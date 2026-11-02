import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def normalize_points(points):
    """
    Centers and scales points to remove translation and scale invariance.
    This stabilizes the optimization.
    """
    pts = points - np.mean(points, axis=0)
    norm_sq = np.mean(np.sum(pts**2, axis=1))
    if norm_sq < 1e-15:
        return pts
    return pts / np.sqrt(norm_sq)

def objective_smooth(x, tau=10.0):
    points = x.reshape(N, D)
    # Normalize to remove translation/scale invariance
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Stable log-sum-exp for soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Stable log-sum-exp for soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def fibonacci_sphere(n_points):
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Icosahedron based (from current)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0
    
    for i in range(10):
        p1 = rng.normal(size=D) * 0.1
        p2 = rng.normal(size=D) * 0.1
        pts = np.vstack([ico, p1, p2])
        candidates.append(pts.flatten())
        
    # 2. Fibonacci sphere (from inspiration 1)
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
        
    # 3. Random points on sphere
    for i in range(15):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
    # 4. Grid-based initialization (from inspiration 1)
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    grid_pts_arr = np.array(final_grid, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    candidates.append(grid_pts_arr.flatten())
    
    # Perturbed grid
    for _ in range(5):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
    
    # 5. Additional: try different random seeds for diversity
    for seed in [1, 7, 13, 99]:
        rng2 = np.random.default_rng(seed)
        pts = rng2.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    # Optimization parameters
    # Increased tau schedule for better approximation of nonsmooth max/min
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [300, 400, 500, 600, 800, 1000]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 5:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                # Use L-BFGS-B for the main optimization
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final high-precision polish with SLSQP
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 800, 'ftol': 1e-16, 'maxls': 100}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Nelder-Mead refinement on raw score (non-smooth)
            if time.time() - start_time < timeout - 3.0:
                try:
                    def raw_obj(x_flat):
                        pts = x_flat.reshape(N, D)
                        s = raw_score(pts)
                        return s if np.isfinite(s) else 1e10
                    
                    res_nm = minimize(
                        raw_obj,
                        x_curr,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res_nm.x is not None:
                        x_curr = res_nm.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # If optimization didn't find anything better, fall back to the best candidate
    if best_points is None:
        # Try all candidates directly
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        
        if best_points is None:
            best_points = grid_pts_arr

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
