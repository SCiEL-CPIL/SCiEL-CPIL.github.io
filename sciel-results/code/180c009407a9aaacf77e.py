"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    P = 40.0  # Higher power for sharper penalty
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        # Pairwise part
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        # Objective value
        sum_pair = np.sum(1.0 / (dists_safe ** P))
        
        # Gradient of pairwise part
        # grad_i = -P * sum_j (d_ij^(-P-2) * (x_i - x_j))
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1)
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - (W @ pts))
        
        # Boundary part
        abs_pts = np.abs(pts)
        b_dists = 0.5 - abs_pts
        min_idx = np.argmin(b_dists, axis=1)
        min_b_vals = np.min(b_dists, axis=1)
        
        min_b_safe = np.where(min_b_vals < 1e-10, 1e-10, min_b_vals)
        sum_bound = np.sum(1.0 / (min_b_safe ** P))
        
        # Gradient of boundary part: P * min_b^(-P-1) * sign(pts[i,k])
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            k = min_idx[i]
            s = np.sign(pts[i, k])
            if s == 0: s = 1.0
            grad_bound[i, k] = P * (min_b_safe[i] ** (-P - 1.0)) * s
        
        total_grad = grad_pair + grad_bound
        total_obj = sum_pair + sum_bound
        
        return total_obj, total_grad.flatten()
    
    def init_grid_5x5x5():
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].flatten()
    
    def init_grid_4x5x5():
        # 4x5x5 = 100 points
        tx = (np.arange(4) + 0.5) / 4 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_5x4x5():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(4) + 0.5) / 4 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_5x5x4():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(4) + 0.5) / 4 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()
    
    def init_clustered():
        # Start with clusters that will expand
        pts = np.random.uniform(-0.4, 0.4, size=(N, 3))
        return pts.flatten()
    
    starts = [
        init_grid_5x5x5(),
        init_grid_4x5x5(),
        init_grid_5x4x5(),
        init_grid_5x5x4(),
        init_grid_5x5x5() + np.random.normal(0, 0.01, size=(N*3,)),
        init_grid_4x5x5() + np.random.normal(0, 0.01, size=(N*3,)),
        init_random(),
        init_random(),
        init_clustered(),
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Track best per start to avoid redundant work
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                           jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < 1e15:
                pts = res.x.reshape(N, 3)
                pts = np.clip(pts, -0.5, 0.5)
                score = calculate_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            continue
    
    # Local refinement: try to improve the best solution with local moves
    if best_points is not None and time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            try:
                x0 = best_points.flatten()
                res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                               jac=True,
                               options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30})
                if res.fun < 1e15:
                    pts = res.x.reshape(N, 3)
                    pts = np.clip(pts, -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                break
    
    if best_points is None:
        best_points = init_grid_5x5x5().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
