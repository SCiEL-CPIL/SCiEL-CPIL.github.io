import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(c):
    # Exact radius calculation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _smooth_obj(x, k):
    c = x.reshape(N, 3)
    # Check bounds (soft penalty if outside, though L-BFGS-B bounds handle this)
    # Calculate pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    # Wall distances
    # d_wall = 0.5 - |c|
    wall_dists = 0.5 - np.abs(c)
    
    # Sum of exponentials
    # To avoid overflow/underflow, we can scale or just be careful.
    # k will be up to maybe 100-500.
    # d is around 0.1-0.2.
    # -k*d is around -10 to -50. exp(-50) is small.
    
    # Mask diagonal for pairs
    dists_flat = dists.flatten()
    # Exclude diagonal (d=0)
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    
    # Wall dists: each point has 3 walls?
    # Actually, distance to wall is min(0.5 - x, 0.5 + x) = 0.5 - |x|.
    # So for each point, we have 3 distances to 3 pairs of walls?
    # No, the distance to the *boundary* is the min of distances to 6 faces.
    # d_boundary_i = min(0.5-c_ix, 0.5+c_ix, 0.5-c_iy, 0.5+c_iy, 0.5-c_iz, 0.5+c_iz)
    # This is equivalent to 0.5 - max(|c_ix|, |c_iy|, |c_iz|)? No.
    # The distance from a point to the cube boundary is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
    # Let's use that.
    
    d_wall_per_point = 0.5 - np.max(np.abs(c), axis=1)
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, d_wall_per_point])
    
    # Log-Sum-Exp
    # min(d) ~ -1/k * log(sum(exp(-k*d)))
    # We want to MINIMIZE the objective, so we want to MAXIMIZE min(d).
    # So we minimize: 1/k * log(sum(exp(-k*d)))
    
    # Numerical stability: shift by max?
    # exp(-k*d). Max d is ~0.2. Min d is ~0.
    # If d is small, exp(-k*d) is large.
    # Let's just compute it.
    
    # To avoid underflow of large sums, we can use scipy.special.logsumexp if available, 
    # but standard numpy is fine for N=50 (1225 + 50 = 1275 terms).
    
    # k * d
    kd = k * all_dists
    # sum(exp(-kd))
    # If kd is large, exp(-kd) is 0.
    # If kd is small, exp(-kd) is ~1.
    
    # We want to minimize the log-sum-exp.
    # log(sum(exp(-kd)))
    
    # Shift for stability:
    max_kd = np.max(kd) # This is the smallest distance term
    # sum(exp(-kd)) = exp(-max_kd) * sum(exp(-(kd - max_kd)))
    # log(...) = -max_kd + log(sum(exp(-(kd - max_kd))))
    
    # Note: kd - max_kd <= 0.
    
    log_sum_exp = -max_kd + np.log(np.sum(np.exp(-(kd - max_kd))))
    
    return log_sum_exp / k

def _generate_fcc_cluster(n):
    # Generate FCC points
    points = []
    max_idx = 10
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k])
                points.append([i+0.5, j, k+0.5])
                points.append([i, j+0.5, k+0.5])
                
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    pts = points[idx[:n]].copy()
    
    # Scale to fit
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        # Scale so that max coord is 0.45 (leaving 0.05 margin for radius)
        scale = 0.45 / max_coord
        pts *= scale
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    c = _generate_fcc_cluster(N)
    
    # Initial radius check
    r = _compute_radius(c)
    
    # Annealing optimization
    k_values = [5, 20, 100, 500]
    
    for k in k_values:
        if time.time() > deadline:
            break
            
        def obj(x, k=k):
            return _smooth_obj(x, k)
            
        try:
            res = minimize(
                obj, 
                c.flatten(), 
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N*3),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                r = r_new
        except Exception:
            pass
            
    # Final polish with exact objective
    if time.time() < deadline:
        def exact_obj(x):
            c_t = x.reshape(N, 3)
            # Penalize out of bounds
            if np.any(c_t < -0.5) or np.any(c_t > 0.5):
                return 1e10
            return -_compute_radius(c_t)
            
        try:
            res = minimize(
                exact_obj, 
                c.flatten(), 
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N*3),
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            c_final = res.x.reshape(N, 3)
            c_final = np.clip(c_final, -0.4999, 0.4999)
            r_final = _compute_radius(c_final)
            if r_final > r:
                c = c_final
                r = r_final
        except Exception:
            pass
            
    return c
