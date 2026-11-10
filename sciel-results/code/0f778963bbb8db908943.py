import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 630.0
    SAFETY_MARGIN = 15.0
    N = 16

    def get_dist_sq(params):
        """Calculate squared pairwise distances."""
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        return dist_sq

    def get_cost_raw(params):
        """
        Calculate (d_max / d_min)^2. Lower is better.
        Used for final evaluation and fallback.
        """
        dist_sq = get_dist_sq(params)
        d_max_sq = np.max(dist_sq)
        d_min_sq = np.min(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return 1e12
            
        return d_max_sq / d_min_sq

    def get_cost_smooth(params, T):
        """
        Smoothed cost function for gradient-based optimization.
        Minimizes -log(d_min_smooth) + log(d_max_smooth)
        which is equivalent to minimizing log(d_max_smooth / d_min_smooth)
        
        d_min_smooth = -T * log(sum(exp(-d_sq/T)))  (Soft Min of distances)
        d_max_smooth = T * log(sum(exp(d_sq/T)))    (Soft Max of distances)
        
        We want to maximize d_min/d_max, so minimize -log(d_min/d_max) = log(d_max) - log(d_min).
        """
        dist_sq = get_dist_sq(params)
        # Mask out inf (diagonal)
        mask = np.isfinite(dist_sq)
        ds = dist_sq[mask]
        
        if len(ds) == 0:
            return 1e12
            
        # Numerical stability for exp
        # Soft Min: -T * log( sum( exp(-d_sq/T) ) )
        # To avoid underflow/overflow, shift by max/min
        neg_ds_over_T = -ds / T
        max_neg = np.max(neg_ds_over_T)
        log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_ds_over_T - max_neg)))
        d_min_smooth = -T * log_sum_exp_neg
        
        # Soft Max: T * log( sum( exp(d_sq/T) ) )
        ds_over_T = ds / T
        max_pos = np.max(ds_over_T)
        log_sum_exp_pos = max_pos + np.log(np.sum(np.exp(ds_over_T - max_pos)))
        d_max_smooth = T * log_sum_exp_pos
        
        if d_min_smooth <= 1e-12:
            return 1e12
            
        # We want to minimize log(d_max) - log(d_min)
        # This is proportional to log(d_max/d_min)
        # Minimizing this is equivalent to minimizing d_max/d_min
        return np.log(d_max_smooth) - np.log(d_min_smooth)

    def smooth_minimize(params, T_start, T_end, num_steps):
        """
        Perform annealing on T, using L-BFGS-B at each step.
        """
        current_params = params.copy()
        T = T_start
        # Geometric progression for T
        ratio = (T_end / T_start) ** (1.0 / num_steps)
        
        for i in range(num_steps):
            T = max(T * ratio, T_end)
            
            # Define objective for this T
            def obj(x):
                return get_cost_smooth(x, T)
                
            try:
                res = minimize(
                    obj, 
                    current_params, 
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < get_cost_smooth(current_params, T):
                    current_params = res.x
            except Exception:
                pass
                
        return current_params

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_polygon = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg_polygon)
    
    # Candidate 2: 4x4 Square Grid
    # Center it
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid = np.column_stack([x.ravel(), y.ravel()])
    candidates.append(grid)
    
    # Candidate 3: Hexagonal Lattice subset
    # Generate a larger hex lattice and pick 16 closest to center
    # Hex lattice vectors
    a = 1.0
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    points_hex = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * v1 + j * v2
            points_hex.append(p)
    points_hex = np.array(points_hex)
    # Find 16 closest to origin
    dists_to_origin = np.sum(points_hex**2, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_grid = points_hex[idx]
    candidates.append(hex_grid)
    
    # Candidate 4: Random
    rng = np.random.default_rng(42)
    rand_pts = rng.uniform(-1, 1, (N, 2))
    candidates.append(rand_pts)

    best_result = None
    best_cost = np.inf

    # 2. Refine each candidate
    # Use the smoothed objective with annealing
    T_start = 1.0
    T_end = 1e-4
    num_anneal_steps = 5
    
    for i, cand in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Initial quick polish with raw cost using Nelder-Mead to get close to a local min of the raw function
        # This helps L-BFGS-B which might struggle if T is too large or small initially
        try:
            # Normalize scale to help convergence
            # The problem is scale invariant, but L-BFGS-B works better with reasonable scales
            # Let's center and scale to unit norm
            cand_centered = cand - np.mean(cand, axis=0)
            norm = np.linalg.norm(cand_centered)
            if norm < 1e-6:
                norm = 1.0
            cand_scaled = cand_centered / norm
            
            # Quick NM polish on scaled version
            nm_res = minimize(
                get_cost_raw, 
                cand_scaled.ravel(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if nm_res.fun < best_cost:
                # Unscale is not necessary for the final answer if we just keep the shape, 
                # but we need to return valid points. 
                # Actually, the score is invariant to scale and translation.
                # So we can just keep the scaled/centered points.
                # However, we must ensure the points are distinct.
                best_cost = nm_res.fun
                best_result = nm_res.x.reshape(N, 2)
        except Exception:
            pass

    # 3. If we have time, run a global search (DE) to find a better basin, then refine
    if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN - 60:
        try:
            # Run DE with a moderate budget
            bounds = [(-1.0, 1.0)] * (N * 2)
            de_res = differential_evolution(
                get_cost_raw, 
                bounds, 
                maxiter=200, 
                popsize=30, 
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False
            )
            
            if de_res.fun < best_cost:
                best_cost = de_res.fun
                best_result = de_res.x.reshape(N, 2)
                
        except Exception as e:
            print(f"DE failed: {e}", file=sys.stderr)

    # 4. Final Deep Polish using L-BFGS-B with Annealing on the best result
    if best_result is not None:
        # Center and scale for numerical stability in L-BFGS-B
        br_centered = best_result - np.mean(best_result, axis=0)
        norm = np.linalg.norm(br_centered)
        if norm < 1e-6:
            norm = 1.0
        br_scaled = br_centered / norm
        
        # Apply smooth annealing
        polished = smooth_minimize(br_scaled.ravel(), T_start, T_end, num_anneal_steps)
        
        # Check if the polished result is better
        polished_cost = get_cost_raw(polished)
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_result = polished.reshape(N, 2)
            
        # One last Nelder-Mead polish on the raw function to clean up any artifacts from smoothing
        try:
            final_nm = minimize(
                get_cost_raw, 
                best_result.ravel(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if final_nm.fun < best_cost:
                best_cost = final_nm.fun
                best_result = final_nm.x.reshape(N, 2)
        except Exception:
            pass

    # Fallback if everything failed
    if best_result is None or not np.all(np.isfinite(best_result)):
        best_result = reg_polygon

    return best_result.reshape(16, 2)
