# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Global search using Differential Evolution in 42D Cartesian space.
    2. The objective function handles centering and normalization to ensure 
       scale and translation invariance.
    3. Refine the best global candidate with a local optimizer (Nelder-Mead).
    4. Strict time management to ensure completion within 600s.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    def process_points(x_flat):
        """
        Reshape flat array to (14, 3), center it, and normalize it.
        Returns the processed points and the metric value.
        """
        pts = x_flat.reshape(n, d)
        
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate distances
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return pts, 0.0
        
        # Normalize to unit max distance (scale invariance)
        pts_norm = pts / d_max
        
        # Recalculate distances for the normalized set to be precise
        # Note: d_min/d_max is invariant to scaling, so we can compute ratio from original
        # d_min / d_max is the same for pts and pts_norm.
        d_min = np.min(dists)
        ratio = d_min / d_max
        metric = ratio ** 2
        
        return pts_norm, metric

    def objective(x_flat):
        """
        Objective function for minimization (we maximize the metric, so minimize negative).
        """
        try:
            if time.time() > deadline - 5:
                return 0.0
            pts, metric = process_points(x_flat)
            return -metric
        except Exception:
            return 0.0

    # Define bounds for Differential Evolution
    # We search in a cube [-1, 1]^42 initially, but since we normalize, 
    # the scale doesn't matter, only the relative positions.
    # However, DE needs bounded space. [-1, 1] is a reasonable start.
    bounds = [(-1.0, 1.0)] * (n * d)

    # Run Differential Evolution
    best_params = None
    best_score = -np.inf

    try:
        # DE settings:
        # popsize=15 (default is 15) -> 15*42 = 630 evaluations per generation
        # maxiter=50 -> 31500 evaluations. 
        # Each eval is fast (pdist on 14 points). 
        # 31500 * ~0.0001s = ~3s. This is very fast.
        # We can increase maxiter or popsize if time permits, but let's start conservative 
        # and use remaining time for refinement or more DE iterations.
        
        # Let's use a more aggressive setting since it's fast.
        # popsize=20, maxiter=100.
        # 20 * 42 * 100 = 84000 evals. Still likely < 10s.
        
        res_de = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            polish=False,  # We will do our own polish
            init='latinhypercube',
            workers=-1     # Use all cores if available
        )
        
        if res_de.success or res_de.fun < 0: # If it found something
            best_params = res_de.x
            best_score = -res_de.fun
        else:
            # Fallback to initial best if DE failed
            best_params = res_de.x
            best_score = -res_de.fun

    except Exception as e:
        # If DE fails completely, fallback to random
        best_params = np.random.uniform(-1, 1, n * d)
        best_score = -objective(best_params)

    # Refine with Nelder-Mead
    # The DE result is likely good but not locally optimal.
    if time.time() < deadline - 10:
        try:
            # Nelder-Mead can struggle in high dimensions if not initialized well,
            # but starting from a good DE point helps.
            # We need to be careful with step size.
            # Let's use L-BFGS-B if we can compute gradients? No, pdist is not differentiable easily at mins.
            # Nelder-Mead is safer.
            
            # We might need to scale the search space for NM.
            # Let's try minimizing the negative metric directly.
            
            # To help NM, we can fix the scale and centering in the objective passed to NM?
            # The `objective` function already does this.
            
            # NM in 42D is slow. Let's limit iterations.
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res_nm.fun < -best_score: # i.e. -metric is smaller (metric larger)
                best_params = res_nm.x
                best_score = -res_nm.fun
                
        except Exception:
            pass

    # If we still have time, do another round of DE from the best point?
    # Or just a few random restarts of NM?
    if time.time() < deadline - 20:
        # Try a few random perturbations of the best solution and refine
        for i in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.1, n * d)
            try:
                res_pert = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res_pert.fun < -best_score:
                    best_params = res_pert.x
                    best_score = -res_pert.fun
            except Exception:
                pass

    # Generate final points from best_params
    points, final_metric = process_points(best_params)

    # Fallback if something went wrong
    if not np.all(np.isfinite(points)) or final_metric == 0:
        # Simple fallback: Icosahedron + 2 poles
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        points = np.vstack([ico_verts, pole1, pole2])
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
