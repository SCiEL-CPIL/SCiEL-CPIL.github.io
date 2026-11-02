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
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * radius

def objective_ratio_smooth(x, tau=10.0):
    """
    Smooth approximation of log(max/min) of squared distances.
    Log-domain formulation is better conditioned.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0) if points coincide
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Work in log domain for better conditioning
    log_dists = np.log(sq_dists)
    
    # Soft max in log domain
    m_max = np.max(log_dists)
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min in log domain
    m_min = np.min(log_dists)
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Return log ratio (equivalent to log of ratio, better conditioned)
    return soft_max_log - soft_min_log

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
    
    # 2. Fibonacci sphere initialization - better spread than random
    fib_pts = fibonacci_sphere(N, radius=2.0)
    candidates.append(fib_pts.flatten())
    
    # 3. Random starts with varied scales
    rng = np.random.default_rng(42)
    num_random_starts = 12
    for _ in range(num_random_starts):
        # Random points in a sphere
        pts = rng.normal(size=(N, D))
        # Normalize to unit sphere roughly to keep scale reasonable
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        scale = rng.uniform(1.5, 3.0)
        pts = pts / norms * scale
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout:
            break
    
    # 4. Add perturbed versions of the grid and fib for diversity
    for _ in range(5):
        perturbed = init_grid + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
        
        if time.time() - start_time > timeout:
            break
    
    for _ in range(3):
        perturbed_fib = fib_pts + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed_fib.flatten())
        
        if time.time() - start_time > timeout:
            break

    # Tau schedule for annealing-like optimization
    tau_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    maxiter_schedule = [150, 200, 300, 400, 300]
    
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
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
                )
                
                if res.x is not None and res.fun < np.inf:
                    x_curr = res.x
            
            # Final high-precision polish with very high tau using L-BFGS-B
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_ratio_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
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
