import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 630.0
    SAFETY_MARGIN = 10.0
    N = 16

    def get_cost_fast(params):
        """Calculate d_max / d_min. Lower is better."""
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        # Exclude diagonal (self-distance)
        np.fill_diagonal(dist_sq, np.inf)
        
        d_max_sq = np.max(dist_sq)
        d_min_sq = np.min(dist_sq)
        
        if d_min_sq <= 1e-12:
            return 1e12
        if d_max_sq <= 1e-12:
            return 1e12
            
        return np.sqrt(d_max_sq / d_min_sq)

    def get_cost_sqp(params, min_dist_sq):
        """
        Objective for SQP: Minimize max_dist_sq + penalty * sum(max(0, min_dist_sq - d_ij)^2)
        """
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        # Exclude diagonal
        np.fill_diagonal(dist_sq, np.inf)
        
        d_max_sq = np.max(dist_sq)
        
        # Constraint violation: distances must be >= min_dist_sq
        # violation = sum((min_dist_sq - d_ij)^2) for d_ij < min_dist_sq
        violations = np.maximum(0, min_dist_sq - dist_sq)
        penalty = np.sum(violations * violations)
        
        # Weighting: we want to balance reducing max distance and satisfying constraints
        # As min_dist_sq increases, the problem becomes harder.
        # If infeasible, penalty dominates.
        return d_max_sq + 100.0 * penalty

    def feasible(params, min_dist_sq):
        """Check if all distances are >= min_dist_sq"""
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        return np.min(dist_sq) >= min_dist_sq

    best_result = None
    best_cost = np.inf
    
    # 1. Initial Global Search with Differential Evolution
    # This finds a good starting point
    bounds = [(-1, 1)] * (N * 2)
    
    try:
        # Run DE for a short time to get a good initial guess
        # popsize=30, maxiter=50 is fast
        de_res = differential_evolution(
            get_cost_fast, 
            bounds, 
            maxiter=50, 
            popsize=30, 
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False
        )
        
        if de_res.fun < best_cost:
            best_cost = de_res.fun
            best_result = de_res.x
            
    except Exception as e:
        print(f"DE failed: {e}")

    # Fallback if DE failed
    if best_result is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)])
        best_cost = get_cost_fast(best_result)

    # 2. Local Polish with Nelder-Mead on the original objective
    # This refines the solution found by DE
    try:
        nm_res = minimize(
            get_cost_fast, 
            best_result, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if nm_res.fun < best_cost:
            best_cost = nm_res.fun
            best_result = nm_res.x
    except Exception as e:
        print(f"NM failed: {e}")

    # 3. Sequential Convex Programming (SCP) via Bisection on min_dist
    # We try to increase the minimum distance while keeping the maximum distance controlled.
    # Actually, the ratio R = (d_min/d_max)^2.
    # Let's fix d_max = 1 (by scaling) and maximize d_min.
    # Or fix d_min = 1 and minimize d_max.
    # The latter is easier to formulate as a convex-like problem (minimize max of convex functions).
    
    # Let's normalize the current best result to have d_max = 1.
    pts = best_result
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    d_max_sq = np.max(dist_sq)
    
    if d_max_sq > 0:
        pts = pts / np.sqrt(d_max_sq)
    
    # Now d_max is 1. We want to maximize d_min.
    # This is equivalent to finding the largest r such that we can place N points
    # in a unit square (or rather, the convex hull is within unit diameter)
    # such that all pairwise distances are >= r.
    
    # We will use bisection on r (min_dist).
    # Low bound: current min_dist
    # High bound: 1.0 (max possible)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    current_min_dist_sq = np.min(dist_sq)
    
    low_r = current_min_dist_sq
    high_r = 1.0
    
    # Bisection search
    # We have limited time, so limit iterations
    time_elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
    
    # Estimate time per SCP iteration
    # Minimize d_max_sq + penalty * ...
    # Let's try 20-30 iterations of bisection if time allows.
    
    max_sqp_iters = 30
    if remaining_time < 5:
        max_sqp_iters = 0
        
    for i in range(max_sqp_iters):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        mid_r = (low_r + high_r) / 2.0
        
        # Solve: Minimize max_dist_sq subject to d_ij >= mid_r
        # Using penalty method: Minimize max_dist_sq + lambda * sum(max(0, mid_r - d_ij)^2)
        # lambda should be large enough to enforce constraints.
        
        def objective_sqp(params):
            return get_cost_sqp(params, mid_r)
            
        try:
            # Use L-BFGS-B or Nelder-Mead?
            # L-BFGS-B is better for smooth approximations, but our penalty is non-smooth at d_ij=mid_r.
            # However, the squared penalty is smooth.
            # Let's use L-BFGS-B for speed.
            res = minimize(
                objective_sqp, 
                pts, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            # Check feasibility of the result
            if feasible(res.x, mid_r):
                # Feasible, try higher r
                low_r = mid_r
                pts = res.x
                # Update best if this configuration is better (higher min_dist for fixed max_dist=1)
                # Since we normalized max_dist to 1, a higher min_dist is strictly better.
                # But wait, the optimization might have changed max_dist.
                # We need to re-normalize and check.
                
                # Actually, if we minimize max_dist_sq, and it's feasible, 
                # the new max_dist_sq might be < 1.
                # We should re-normalize to keep the "unit diameter" assumption for the next step?
                # Or just track the actual ratio.
                
                # Let's just track the actual best result based on the original metric.
                new_cost = get_cost_fast(res.x)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_result = res.x
                    
            else:
                # Infeasible, try lower r
                high_r = mid_r
                
        except Exception as e:
            # If solver fails, assume infeasible?
            high_r = mid_r

    # Final Polish on the best result found
    if best_result is not None:
        time_elapsed = time.time() - start_time
        polish_budget = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
        
        if polish_budget > 5:
            try:
                nm_res = minimize(
                    get_cost_fast, 
                    best_result, 
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                if nm_res.fun < best_cost:
                    best_cost = nm_res.fun
                    best_result = nm_res.x
            except Exception as e:
                print(f"Final NM failed: {e}")

    # Final check
    if not np.all(np.isfinite(best_result)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_result.reshape(16, 2)
