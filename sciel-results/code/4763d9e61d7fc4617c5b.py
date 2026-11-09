import numpy as np
import time
from scipy.optimize import minimize

def calculate_scores(points):
    """
    Calculate the vertex scores for a given polygon.
    Returns an array of 20 scores.
    """
    n = len(points)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        # Distances from i to all others, exclude self
        d = np.delete(dists[i], i)
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check for repeated points (dist < 0.05)
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
            continue
            
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(diffs)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17: # n-3? 19-1=18 indices. 17 is second to last gap?
            # Indices 0..18. Gaps 0..17.
            # If k=17, gap is d_18 - d_17.
            # Rule: d_15..d_18
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def min_vertex_score(radii):
    """
    radii: array of 20 radii.
    Returns the minimum vertex score.
    """
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Normalize: subtract centroid, divide by mean dist to centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist
    
    # Check convexity
    # Compute cross products of consecutive edges
    # Edge i: p_{i+1} - p_i
    # Cross product z-component: (p_{i+1}-p_i) x (p_{i+2}-p_{i+1})
    # (x_{i+1}-x_i)*(y_{i+2}-y_{i+1}) - (y_{i+1}-y_i)*(x_{i+2}-x_{i+1})
    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    
    e1 = p_next - p
    e2 = p_next2 - p_next
    
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Strictly convex: all cross products non-zero and same sign
    if np.any(cross == 0):
        return 0.0
    if np.any(cross > 0) and np.any(cross < 0):
        return 0.0
        
    # Also check no repeated points
    if np.min(np.linalg.norm(p_next - p, axis=1)) < 1e-9:
         return 0.0
         
    scores = calculate_scores(points)
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 40s safety margin for 630s limit
    
    n = 20
    # Initial guess: regular polygon
    x0 = np.ones(n)
    
    # We want to maximize min_vertex_score
    # scipy.minimize minimizes, so we negate
    def objective(r):
        # Ensure radii are positive to maintain convexity/validity
        r = np.maximum(r, 0.1)
        return -min_vertex_score(r)
        
    best_result = None
    best_score = -1.0
    
    def try_optimize(x_init, method='Nelder-Mead', maxiter=5000, xatol=1e-10, fatol=1e-10):
        nonlocal best_result, best_score
        try:
            if method == 'Nelder-Mead':
                res = minimize(objective, x_init, method=method,
                               options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
            elif method == 'Powell':
                res = minimize(objective, x_init, method=method,
                               options={'maxiter': maxiter, 'xtol': xatol, 'ftol': fatol})
            else:
                res = minimize(objective, x_init, method=method,
                               options={'maxiter': maxiter})
            if res.fun < best_score:
                best_score = -res.fun
                best_result = res.x
        except Exception:
            pass
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    try:
        # Restart 1: Regular
        try_optimize(x0, 'Nelder-Mead', 5000, 1e-10, 1e-10)
        
        # Restart 2: Wavy 4 bumps
        if time.time() < deadline - 80:
            x1 = np.maximum(1.0 + 0.15 * np.cos(4*angles), 0.1)
            try_optimize(x1, 'Nelder-Mead', 5000, 1e-10, 1e-10)
        
        # Restart 3: Wavy 5 bumps
        if time.time() < deadline - 80:
            x3 = np.maximum(1.0 + 0.15 * np.cos(5*angles), 0.1)
            try_optimize(x3, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 4: Wavy 2 bumps
        if time.time() < deadline - 60:
            x4 = np.maximum(1.0 + 0.15 * np.cos(2*angles), 0.1)
            try_optimize(x4, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 5: Wavy 3 bumps
        if time.time() < deadline - 60:
            x5 = np.maximum(1.0 + 0.15 * np.cos(3*angles), 0.1)
            try_optimize(x5, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 6: Wavy 6 bumps
        if time.time() < deadline - 60:
            x6 = np.maximum(1.0 + 0.15 * np.cos(6*angles), 0.1)
            try_optimize(x6, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 7: Random perturbation seed 42
        if time.time() < deadline - 50:
            np.random.seed(42)
            x7 = np.maximum(1.0 + 0.08 * np.random.randn(n), 0.1)
            try_optimize(x7, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 8: Random perturbation seed 123
        if time.time() < deadline - 50:
            np.random.seed(123)
            x8 = np.maximum(1.0 + 0.08 * np.random.randn(n), 0.1)
            try_optimize(x8, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 9: Random perturbation seed 777
        if time.time() < deadline - 50:
            np.random.seed(777)
            x9 = np.maximum(1.0 + 0.08 * np.random.randn(n), 0.1)
            try_optimize(x9, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 10: Larger amplitude wavy 4
        if time.time() < deadline - 40:
            x10 = np.maximum(1.0 + 0.25 * np.cos(4*angles), 0.1)
            try_optimize(x10, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 11: Larger amplitude wavy 5
        if time.time() < deadline - 40:
            x11 = np.maximum(1.0 + 0.25 * np.cos(5*angles), 0.1)
            try_optimize(x11, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 12: Alternating pattern
        if time.time() < deadline - 30:
            x12 = np.ones(n)
            x12[::2] = 1.1
            x12[1::2] = 0.9
            try_optimize(x12, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 13: Alternating with offset
        if time.time() < deadline - 30:
            x13 = np.ones(n)
            x13[1::2] = 1.1
            x13[0::2] = 0.9
            try_optimize(x13, 'Nelder-Mead', 5000, 1e-10, 1e-10)

        # Restart 14: Powell method on best found so far
        if best_result is not None and time.time() < deadline - 30:
            try_optimize(best_result, 'Powell', 2000, 1e-10, 1e-10)

        # Restart 15: Final polish with Nelder-Mead on best
        if best_result is not None and time.time() < deadline - 10:
            try_optimize(best_result, 'Nelder-Mead', 3000, 1e-12, 1e-12)

    except Exception:
        pass
        
    if best_result is None:
        best_result = x0
        
    # Construct final polygon
    x = best_result * np.cos(angles)
    y = best_result * np.sin(angles)
    polygon = np.column_stack((x, y))
    
    # Final Normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
        
    return polygon
