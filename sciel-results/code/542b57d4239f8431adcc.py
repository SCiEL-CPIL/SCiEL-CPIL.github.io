# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # 10s margin
    
    # Initial guess: regular polygon
    # Parameterization: x -> gaps g > 0, sum(g)=2*pi
    # g_i = 2*pi * exp(x_i) / sum(exp(x))
    
    x0 = np.zeros(n)
    
    def get_gaps(x):
        exp_x = np.exp(x)
        return 2.0 * np.pi * exp_x / np.sum(exp_x)

    def compute_min_score(g):
        # g: shape (n,)
        angles = np.cumsum(g)
        # Vertices at angles[0], ..., angles[n-1]
        # angles[n-1] is sum(g[0]..g[n-2])
        # The last gap g[n-1] closes the loop to 2*pi.
        # So we use the first n angles.
        pts = np.column_stack((np.cos(angles[:n]), np.sin(angles[:n])))
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        min_score = 1.0
        for i in range(n):
            d = dists[i].copy()
            d[i] = np.inf
            d_sorted = np.sort(d)
            
            if np.any(d_sorted < 0.05):
                return 0.0
                
            gaps = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(gaps)
            
            if k == 0:
                ds = d_sorted[0:4]
            elif k == 17:
                ds = d_sorted[15:19]
            else:
                ds = d_sorted[k-1:k+3]
                
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                return 0.0
                
            ratios = np.maximum(ds / D, D / ds)
            raw = np.mean(ratios)
            s_i = min(1.0, 1.0 / raw)
            if s_i < min_score:
                min_score = s_i
        return min_score

    def objective(x):
        if time.time() > deadline:
            return 1e6
        g = get_gaps(x)
        # We want to maximize the min score.
        # Use soft-min for smoothness.
        # But compute_min_score returns the HARD min.
        # To use soft-min, we need individual scores.
        
        # Let's modify to return scores array for soft-min.
        angles = np.cumsum(g)
        pts = np.column_stack((np.cos(angles[:n]), np.sin(angles[:n])))
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        scores = np.zeros(n)
        for i in range(n):
            d = dists[i].copy()
            d[i] = np.inf
            d_sorted = np.sort(d)
            
            if np.any(d_sorted < 0.05):
                scores[i] = 0.0
                continue
                
            gaps = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(gaps)
            
            if k == 0:
                ds = d_sorted[0:4]
            elif k == 17:
                ds = d_sorted[15:19]
            else:
                ds = d_sorted[k-1:k+3]
                
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                scores[i] = 0.0
                continue
                
            ratios = np.maximum(ds / D, D / ds)
            raw = np.mean(ratios)
            scores[i] = min(1.0, 1.0 / raw)
            
        T = 20.0
        max_s = np.max(scores)
        exp_terms = np.exp(-T * (scores - max_s))
        soft_min = max_s - (1.0/T) * np.log(np.sum(exp_terms))
        
        return -soft_min

    best_score = 0.0
    best_x = x0
    
    # Evaluate initial
    g0 = get_gaps(x0)
    # Compute hard min for initial
    angles0 = np.cumsum(g0)
    pts0 = np.column_stack((np.cos(angles0[:n]), np.sin(angles0[:n])))
    diff0 = pts0[:, np.newaxis, :] - pts0[np.newaxis, :, :]
    dists0 = np.sqrt(np.sum(diff0**2, axis=2))
    min_s0 = 1.0
    for i in range(n):
        d = dists0[i].copy()
        d[i] = np.inf
        d_sorted = np.sort(d)
        if np.any(d_sorted < 0.05):
            min_s0 = 0.0
            break
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        if k == 0: ds = d_sorted[0:4]
        elif k == 17: ds = d_sorted[15:19]
        else: ds = d_sorted[k-1:k+3]
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            min_s0 = 0.0
            break
        ratios = np.maximum(ds / D, D / ds)
        raw = np.mean(ratios)
        s_i = min(1.0, 1.0 / raw)
        if s_i < min_s0: min_s0 = s_i
    best_score = min_s0

    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if time.time() < deadline - 10:
             # Evaluate result
             g_res = get_gaps(res.x)
             angles_res = np.cumsum(g_res)
             pts_res = np.column_stack((np.cos(angles_res[:n]), np.sin(angles_res[:n])))
             diff_res = pts_res[:, np.newaxis, :] - pts_res[np.newaxis, :, :]
             dists_res = np.sqrt(np.sum(diff_res**2, axis=2))
             
             min_s_res = 1.0
             for i in range(n):
                 d = dists_res[i].copy()
                 d[i] = np.inf
                 d_sorted = np.sort(d)
                 if np.any(d_sorted < 0.05):
                     min_s_res = 0.0
                     break
                 gaps = d_sorted[1:] - d_sorted[:-1]
                 k = np.argmin(gaps)
                 if k == 0: ds = d_sorted[0:4]
                 elif k == 17: ds = d_sorted[15:19]
                 else: ds = d_sorted[k-1:k+3]
                 D = (d_sorted[k] + d_sorted[k+1]) / 2.0
                 if D < 0.1:
                     min_s_res = 0.0
                     break
                 ratios = np.maximum(ds / D, D / ds)
                 raw = np.mean(ratios)
                 s_i = min(1.0, 1.0 / raw)
                 if s_i < min_s_res: min_s_res = s_i
                 
             if min_s_res > best_score:
                 best_score = min_s_res
                 best_x = res.x
                 
    except Exception:
        pass

    # Final conversion
    g_best = get_gaps(best_x)
    angles_best = np.cumsum(g_best)
    x = np.cos(angles_best[:n])
    y = np.sin(angles_best[:n])
    polygon = np.column_stack((x, y))
    
    return polygon

# EVOLVE-BLOCK-END
