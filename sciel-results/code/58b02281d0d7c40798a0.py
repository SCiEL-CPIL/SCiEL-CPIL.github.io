# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 630s. We aim to finish by 600s.
    deadline = start_time + 590.0
    
    n = 16

    def score(pts):
        """Compute the true metric R = (d_min / d_max)^2"""
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_soft_ratio_flat(flat_pts, beta=1.0):
        """
        Smooth surrogate for the objective function.
        Uses log-sum-exp to approximate min and max smoothly.
        """
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        
        # Soft min: -beta * log(sum(exp(-d/beta)))
        # To avoid overflow, we can shift d, but pdist values are usually small.
        # Let's use a stable version.
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        
        # If beta is very small, exp might overflow. 
        # Standard stable log-sum-exp for min:
        # soft_min = d_min_val - beta * log(sum(exp(-(d - d_min_val)/beta)))
        # soft_max = d_max_val + beta * log(sum(exp((d - d_max_val)/beta)))
        
        # For optimization, we want to maximize (soft_min / soft_max)^2
        # So we minimize - (soft_min / soft_max)^2
        
        # Shift for numerical stability
        # soft_min
        exp_neg = np.exp(-(d - d_min_val) / beta)
        soft_min = d_min_val - beta * np.log(np.sum(exp_neg))
        
        # soft_max
        exp_pos = np.exp((d - d_max_val) / beta)
        soft_max = d_max_val + beta * np.log(np.sum(exp_pos))
        
        if soft_max <= 1e-12:
            return 0.0
            
        return -(soft_min / soft_max) ** 2

    def neg_ratio_flat(flat_pts):
        """Nonsmooth objective for final polishing"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def make_hexagonal_initial():
        """Create a hexagonal lattice-based initial configuration."""
        s = np.sqrt(3) / 2
        candidates = []
        # Generate a grid of lattice points
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + j * 0.5
                y = j * s
                candidates.append([x, y])
        candidates = np.array(candidates)
        candidates -= np.mean(candidates, axis=0)
        
        # Greedy farthest-point sampling to select 16 points
        centroid = np.mean(candidates, axis=0)
        dists_centroid = np.linalg.norm(candidates - centroid, axis=1)
        
        selected = [int(np.argmax(dists_centroid))]
        remaining = list(set(range(len(candidates))) - set(selected))
        
        while len(selected) < n and remaining:
            sel_pts = candidates[selected]
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                pt = candidates[idx]
                min_d = np.min(np.linalg.norm(sel_pts - pt, axis=1))
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        initial_pts = candidates[selected]
        initial_pts -= np.mean(initial_pts, axis=0)
        return initial_pts

    def make_ring_initial():
        """Create concentric ring configuration."""
        pts = []
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 9 points (approx)
        r2 = 2.0
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    # 1. Generate diverse initial populations
    rng = np.random.RandomState(42)
    
    # We will run Differential Evolution (DE) on the soft surrogate.
    # DE requires bounds. Based on the problem scale, points should be within [-3, 3] roughly.
    # We can scale the solution later, so bounds just need to be wide enough.
    bounds = [(-3.0, 3.0)] * (n * 2)
    
    best_pts = None
    best_val = -np.inf
    
    # Strategy: 
    # 1. Run DE on soft surrogate with annealed beta.
    # 2. Polish the best result with L-BFGS-B on the true nonsmooth objective.
    
    # We perform a few rounds of DE with different beta values to balance exploration/exploitation.
    # High beta = smoother, more global. Low beta = sharper, more local.
    betas = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    # Initial population for DE can be seeded with our structured guesses to help convergence
    # However, DE creates its own population. We can just let it run.
    # To save time, we limit maxiter.
    
    for beta in betas:
        if time.time() > deadline - 60:
            break
            
        # Run Differential Evolution
        # We use a smaller population and fewer iterations to fit in the time budget
        # popsize=20, maxiter=100 is reasonable for 32 dimensions.
        try:
            res = differential_evolution(
                lambda x: neg_soft_ratio_flat(x, beta=beta),
                bounds=bounds,
                seed=42,
                maxiter=150,
                popsize=25,
                tol=1e-5,
                polish=False, # We will do our own polishing
                init='sobol' # Sobol sequence for better initial spread
            )
            
            if res.success or res.fun < -best_val:
                candidate_pts = res.x.reshape(n, 2)
                # Center it
                candidate_pts -= np.mean(candidate_pts, axis=0)
                
                # Evaluate on true objective
                val = score(candidate_pts)
                if val > best_val:
                    best_val = val
                    best_pts = candidate_pts
                    
        except Exception as e:
            # If DE fails, skip
            pass

    # If DE didn't find anything or failed, fall back to structured starts
    if best_pts is None:
        best_pts = make_hexagonal_initial()
        best_val = score(best_pts)

    # 2. Refinement Phase
    # Use L-BFGS-B on the TRUE nonsmooth objective to polish the result.
    # This is crucial because the soft surrogate is only an approximation.
    
    if best_pts is not None and time.time() < deadline - 30:
        # Run L-BFGS-B from the best DE solution
        # We might run it multiple times with slightly perturbed starts to escape local minima of the nonsmooth function
        final_best_pts = best_pts
        final_best_val = best_val
        
        # Try a few local optimizations from the best point and its perturbations
        local_starts = [best_pts]
        for i in range(5):
            perturbed = best_pts + rng.randn(n, 2) * 0.05
            perturbed -= np.mean(perturbed, axis=0)
            local_starts.append(perturbed)
            
        for start in local_starts:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    neg_ratio_flat,
                    start.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                opt_pts = res.x.reshape(n, 2)
                opt_pts -= np.mean(opt_pts, axis=0)
                val = score(opt_pts)
                
                if val > final_best_val:
                    final_best_val = val
                    final_best_pts = opt_pts
                    
            except Exception:
                continue
        
        best_pts = final_best_pts
        best_val = final_best_val

    # 3. Final Safety Check
    # Ensure the output is valid and finite
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = make_hexagonal_initial()
    
    # One last quick polish if we have time
    if time.time() < deadline - 5:
        try:
            res = minimize(
                neg_ratio_flat,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            final_pts = res.x.reshape(n, 2)
            final_pts -= np.mean(final_pts, axis=0)
            if score(final_pts) > score(best_pts):
                best_pts = final_pts
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
