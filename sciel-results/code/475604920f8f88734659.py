import numpy as np
import time
from scipy.optimize import minimize
import itertools
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    # points shape (N, D)
    # Use broadcasting to compute all pairwise squared distances
    # diff shape (N, N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    # We only need the upper triangle to avoid duplicates and zeros
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_ratio_smooth(x, tau=10.0):
    """
    Smooth approximation of the ratio of max to min squared distance.
    Uses a better-conditioned formulation.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0) if points coincide
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    # Compute soft max and soft min using log-sum-exp trick
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Return the ratio directly (not log) for better gradient behavior
    # Add small epsilon to avoid division by zero
    return soft_max / (soft_min + 1e-10)

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Initial candidates
    # 1. The grid from the current solution
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    candidates = [init_grid.flatten()]
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    num_random_starts = 25
    for _ in range(num_random_starts):
        # Random points in a sphere
        pts = rng.normal(size=(N, D))
        # Normalize to unit sphere roughly to keep scale reasonable
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout:
            break
    
    # 3. Add perturbed versions of the grid for diversity
    for _ in range(5):
        perturbed = init_grid + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
        
        if time.time() - start_time > timeout:
            break

    # Tau schedule for annealing-like optimization
    tau_schedule = [2.0, 10.0, 50.0, 200.0]
    maxiter_schedule = [150, 200, 300, 400]
    
    # Optimization loop
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout:
                    break
                
                obj = partial(objective_ratio_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                # Use L-BFGS-B
                res = minimize(
                    obj, 
                    x_curr, 
                    method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.x is not None and res.fun < np.inf:
                    x_curr = res.x
            
            # Final high-precision polish with SLSQP
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_ratio_smooth, tau=500.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 200, 'ftol': 1e-14}
                    )
                    if res_polish.x is not None and res_polish.fun < np.inf:
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score for the result
            pts = x_curr.reshape(N, D)
            sq_dists = get_pairwise_sq_dists(pts)
            if np.min(sq_dists) > 1e-10:
                true_score = (np.max(sq_dists) / np.min(sq_dists))
                if true_score < best_score:
                    best_score = true_score
                    best_points = pts
                    
        except Exception:
            pass

    # If optimization didn't find anything better, fall back to the grid
    if best_points is None:
        best_points = init_grid
        sq_dists = get_pairwise_sq_dists(best_points)
        best_score = np.max(sq_dists) / np.min(sq_dists)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
