# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize, SLSQP

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Formulate as a constrained optimization problem where all pairwise distances are forced to be equal (or as close as possible).
    2. Use a penalty method combined with SLSQP to handle the non-smooth min/max objective by approximating it with a smooth penalty term that encourages distance uniformity.
    3. Use Differential Evolution to find a good starting point for the local SLSQP optimizer, as SLSQP is sensitive to initial conditions.
    4. Multiple restarts with different random seeds to escape local minima.
    5. Strict time management to ensure completion within 600s.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    def evaluate_config(points):
        """
        Takes an array of shape (14, 3), centers, normalizes, and returns the metric.
        Returns (points_normalized, metric_value).
        """
        if not np.all(np.isfinite(points)):
            return None, 0.0
            
        pts = points.copy()
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Check for coincident points (d_min = 0)
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return None, 0.0
            
        d_min = np.min(dists)
        
        # Normalize to unit max distance
        pts_norm = pts / d_max
        
        ratio = d_min / d_max
        metric = ratio ** 2
        
        return pts_norm, metric

    def penalty_objective(params, penalty_weight):
        """
        Objective function for SLSQP.
        Uses a penalty method to approximate the min/max ratio.
        We want to maximize d_min/d_max.
        Equivalent to minimizing (d_max/d_min) or maximizing (d_min/d_max)^2.
        
        A common approach is to minimize the variance of pairwise distances,
        subject to the constraint that the points are not coincident.
        
        Alternatively, we can use a smooth approximation of the min function.
        min(d_i) ≈ -logsumexp(-d_i / t) * t
        max(d_i) ≈ logsumexp(d_i / t) * t
        
        Let's try minimizing: logsumexp(d_i / t) / (-logsumexp(-d_i / t))
        which is d_max / d_min. We want to minimize this.
        
        However, SLSQP is a local optimizer. Let's define a simpler objective:
        Minimize the variance of the pairwise distances.
        If all distances are equal, the ratio is 1.
        """
        if time.time() > deadline - 10:
            return 0.0
            
        try:
            pts = params.reshape(n, d)
            # Center
            pts = pts - np.mean(pts, axis=0)
            
            dists = pdist(pts)
            if np.any(dists < 1e-10):
                return 1e10
                
            # Normalize
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 1e10
            dists_norm = dists / d_max
            
            # We want to maximize the minimum distance.
            # Let's minimize the sum of squared differences from the mean distance.
            # This encourages all distances to be similar.
            mean_d = np.mean(dists_norm)
            variance = np.sum((dists_norm - mean_d) ** 2)
            
            # Add a penalty if the minimum distance is too small
            d_min_norm = np.min(dists_norm)
            penalty = 0
            if d_min_norm < 0.5:
                penalty = (0.5 - d_min_norm) ** 2 * penalty_weight
                
            return variance + penalty
        except Exception:
            return 1e10

    def objective_de(params):
        """
        Objective function for Differential Evolution.
        Returns negative metric (to minimize).
        """
        if time.time() > deadline - 10:
            return 0.0
            
        try:
            pts = params.reshape(n, d)
            _, metric = evaluate_config(pts)
            return -metric
        except Exception:
            return 0.0

    best_points = None
    best_score = 0.0

    # --- Phase 1: Global Search with DE ---
    # Use a lower-dimensional parameterization? No, let's try full 42D first with DE.
    # But 42D is high for DE. Let's try a symmetry-based parameterization again, but different.
    # Or, we can use a coarser grid for DE.
    
    # Let's try a different symmetry: 14 points can be seen as 2 sets of 7.
    # Or maybe a truncated octahedron?
    
    # Let's stick to the full 42D space but use a smarter initialization.
    # Initialize with random points on a sphere.
    
    if time.time() < deadline - 120:
        try:
            # DE in 42D is expensive. Let's limit the population and iterations.
            # Bounds: [-2, 2] for each coordinate.
            bounds = [(-2.0, 2.0)] * (n * d)
            
            res_de = differential_evolution(
                objective_de,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='latinhypercube'
            )
            
            params_best = res_de.x
            pts_best = params_best.reshape(n, d)
            pts_norm, score = evaluate_config(pts_best)
            
            if score > best_score:
                best_score = score
                best_points = pts_norm
                
        except Exception:
            pass

    # --- Phase 2: Local Refinement with SLSQP using Penalty Method ---
    # Try multiple restarts from the DE result and random points.
    
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline - 60:
            break
            
        try:
            if i == 0 and best_points is not None:
                # Start from the DE result
                x0 = best_points.flatten()
            else:
                # Random start on a sphere
                x0 = np.random.randn(n * d)
                x0 /= np.linalg.norm(x0)
                x0 *= 2.0 # Scale up
            
            # SLSQP with penalty method
            # We need to define the objective and constraints for SLSQP.
            # Objective: Minimize variance of distances + penalty for small min distance.
            # Constraints: None explicitly, but we can add a constraint that d_min > epsilon.
            
            def slsqp_obj(p):
                return penalty_objective(p, penalty_weight=10.0)
            
            # Constraints: d_min > 0.1 (after normalization)
            # This is tricky because normalization happens inside.
            # Let's just use the penalty method without explicit constraints for simplicity.
            
            res_slsqp = minimize(
                slsqp_obj,
                x0,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            pts_slsqp = res_slsqp.x.reshape(n, d)
            pts_norm, score = evaluate_config(pts_slsqp)
            
            if score > best_score:
                best_score = score
                best_points = pts_norm
                
        except Exception:
            pass

    # --- Phase 3: Nelder-Mead Refinement on the Best Candidate ---
    if best_points is not None and time.time() < deadline - 30:
        try:
            x0 = best_points.flatten()
            
            def nm_obj(p):
                if time.time() > deadline - 10:
                    return 0.0
                pts = p.reshape(n, d)
                _, metric = evaluate_config(pts)
                return -metric
            
            res_nm = minimize(
                nm_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            pts_nm = res_nm.x.reshape(n, d)
            pts_norm, score = evaluate_config(pts_nm)
            
            if score > best_score:
                best_score = score
                best_points = pts_norm
                
        except Exception:
            pass

    # Fallback if no valid points found
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Simple fallback: Icosahedron + 2 poles (normalized)
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
        best_points = points

    return best_points

# EVOLVE-BLOCK-END
