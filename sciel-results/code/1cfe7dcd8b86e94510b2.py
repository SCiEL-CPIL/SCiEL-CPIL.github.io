# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _is_valid(circles, tol=1e-9):
    """Check if circles are valid: inside unit square and disjoint."""
    n = len(circles)
    if n != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    # Vectorized pairwise distance check
    coords = np.column_stack([x, y])
    dist_matrix = cdist(coords, coords)
    r_sum = r[:, None] + r[None, :]
    
    # Set diagonal to 0 to ignore self-constraint
    np.fill_diagonal(dist_matrix, 0)
    np.fill_diagonal(r_sum, 0)
    
    # Check if any distance is less than sum of radii
    if np.any(dist_matrix < r_sum - tol):
        return False
        
    return True

def _score(circles):
    """Calculate the score (sum of radii) if valid, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0

def _make_grid_initial(n_circles=N):
    """Create a grid initial guess."""
    # 2x13 grid
    positions = []
    for i in range(2):
        for j in range(13):
            x = (j + 0.5) / 13.0
            y = (i + 0.5) / 2.0
            positions.append([x, y])
            
    circles = np.array(positions + [[0.03] * 2 for _ in range(len(positions))])
    circles[:, 2] = 0.03
    return circles

def _smooth_objective(params, n_circles):
    """
    Objective function for optimization.
    Minimize: -sum(r) + penalty(overlaps) + penalty(boundaries)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for negative radius
    pen_r = np.sum(np.maximum(0, -r) ** 2)
    
    # Boundary penalties
    pen_bound = np.sum(np.maximum(0, r - x) ** 2) + \
                np.sum(np.maximum(0, x + r - 1) ** 2) + \
                np.sum(np.maximum(0, r - y) ** 2) + \
                np.sum(np.maximum(0, y + r - 1) ** 2)
    
    # Overlap penalties
    coords = np.column_stack([x, y])
    dist_matrix = cdist(coords, coords)
    r_sum = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
    violations = np.maximum(0, r_sum - dist_matrix)[mask]
    pen_overlap = np.sum(violations ** 2)
    
    weight = 100.0
    total_penalty = weight * (pen_r + pen_bound + pen_overlap)
    
    return -np.sum(r) + total_penalty

def _optimize_smooth(circles0, max_iters=2000, ftol=1e-12):
    """
    Optimize using L-BFGS-B with smooth objective.
    """
    n = len(circles0)
    x0 = circles0.flatten()
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    try:
        result = minimize(_smooth_objective, x0, args=(n,), method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': max_iters, 'ftol': ftol, 'gtol': 1e-8})
        
        res = result.x.reshape(n, 3)
        return res
    except Exception:
        return circles0

def _polish(circles, time_limit=5.0):
    """
    High-precision polish: 
    1. For each circle, try to increase radius as much as possible.
    2. Re-optimize positions to accommodate larger radii.
    """
    current = circles.copy()
    start_time = time.time()
    
    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Expand radii
        for i in range(N):
            r_low = current[i, 2]
            r_high = 0.5
            x, y = current[i, 0], current[i, 1]
            r_bound = min(x, 1-x, y, 1-y)
            r_high = min(r_high, r_bound)
            
            if r_high <= r_low + 1e-6:
                continue
                
            for _ in range(20):
                r_mid = (r_low + r_high) / 2
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((x - current[j, 0])**2 + (y - current[j, 1])**2)
                    if dist < r_mid + current[j, 2]:
                        valid = False
                        break
                if valid:
                    r_low = r_mid
                else:
                    r_high = r_mid
                    
            current[i, 2] = r_low
            
        # 2. Re-optimize positions
        if time.time() - start_time > time_limit:
            break
        current = _optimize_smooth(current, max_iters=200, ftol=1e-10)
        
    return current

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_result = None
    best_score = -np.inf
    
    c0 = _make_grid_initial(N)
    
    c1 = _optimize_smooth(c0, max_iters=2000, ftol=1e-12)
    if _is_valid(c1):
        s1 = np.sum(c1[:, 2])
        if s1 > best_score:
            best_score = s1
            best_result = c1.copy()
    else:
        if _is_valid(c0):
            best_result = c0.copy()
            best_score = np.sum(c0[:, 2])
        else:
            c_fallback = np.array([[(i % 5 + 0.5)/5.0, (i // 5 + 0.5)/5.0, 0.01] for i in range(25)] + [[0.5, 0.5, 0.01]])
            if _is_valid(c_fallback):
                best_result = c_fallback
                best_score = np.sum(c_fallback[:, 2])
            else:
                return c_fallback

    if time.time() < deadline - 10:
        c2 = _polish(c1, time_limit=30.0)
        if _is_valid(c2):
            s2 = np.sum(c2[:, 2])
            if s2 > best_score:
                best_score = s2
                best_result = c2.copy()

    if time.time() < deadline - 20:
        c3 = _optimize_smooth(best_result, max_iters=1000, ftol=1e-14)
        if _is_valid(c3):
            s3 = np.sum(c3[:, 2])
            if s3 > best_score:
                best_score = s3
                best_result = c3.copy()

    if time.time() < deadline - 5:
        c4 = _polish(best_result, time_limit=10.0)
        if _is_valid(c4):
            s4 = np.sum(c4[:, 2])
            if s4 > best_score:
                best_score = s4
                best_result = c4.copy()

    if best_result is not None and _is_valid(best_result):
        return best_result
    else:
        return np.array([[(i % 5 + 0.5)/5.0, (i // 5 + 0.5)/5.0, 0.01] for i in range(25)] + [[0.5, 0.5, 0.01]])
# EVOLVE-BLOCK-END
