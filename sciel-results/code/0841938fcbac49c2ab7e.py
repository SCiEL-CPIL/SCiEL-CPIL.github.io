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
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0) if points coincide
    # Add a small epsilon
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max: (1/tau) * log(sum(exp(tau * x)))
    # Soft min: -(1/tau) * log(sum(exp(-tau * x)))
    
    # To prevent overflow in exp, we can shift by max/min
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max calculation
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min calculation
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # We want to minimize the ratio. 
    # If we minimize log(ratio) = log(soft_max) - log(soft_min), it's scale invariant.
    # However, if soft_min is close to 0, log is very negative.
    # Let's just return the ratio itself, but the gradient of ratio is tricky at 0.
    # Minimizing log(soft_max) - log(soft_min) is better conditioned.
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

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
    num_random_starts = 30
    for _ in range(num_random_starts):
        # Random points in a sphere
        pts = rng.normal(size=(N, D))
        # Normalize to unit sphere roughly to keep scale reasonable
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout:
            break

    # Tau schedule for annealing-like optimization
    tau_schedule = [1.0, 5.0, 20.0, 100.0]
    maxiter_schedule = [200, 300, 500, 500]
    
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
                    options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    x_curr = res.x
            
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
