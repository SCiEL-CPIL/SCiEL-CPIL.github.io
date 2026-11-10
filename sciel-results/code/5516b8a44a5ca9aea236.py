import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, n=64, n_heights=20000):
    """
    Compute the union area of the triangles.
    positions: array of shape (n,)
    """
    if positions.shape != (n,):
        positions = positions.reshape(n)
        
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(overlaps, axis=0)
    
    area = np.mean(union_lengths)
    return area

def _subgradient(positions, n=64, n_heights=20000):
    """
    Compute a subgradient of the area.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    sg = np.zeros((n, n_heights))
    
    left_idx = sort_idx[:-1, :]
    right_idx = sort_idx[1:, :]
    
    mask = diffs < w[np.newaxis, :]
    
    rows, cols = np.where(mask)
    
    if len(rows) > 0:
        right_orig = right_idx[rows, cols]
        left_orig = left_idx[rows, cols]
        
        np.add.at(sg, (right_orig, cols), 1.0)
        np.add.at(sg, (left_orig, cols), -1.0)
        
    gradient = np.mean(sg, axis=1)
    
    return gradient

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the positions of the 64 triangles to minimize the union area.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    a = (np.arange(n) + 1) / n
    
    # Initial guesses
    # The theoretical optimum for a single pivot is x_i = -a_i * pivot_y
    # For a fan, pivot is often around 0.5
    x_init_1 = -0.5 * a
    x_init_2 = np.zeros(n)
    x_init_3 = -a
    np.random.seed(42)
    x_init_4 = np.random.normal(0, 0.5, n)
    
    starts = [x_init_1, x_init_2, x_init_3, x_init_4]
    
    best_pos = None
    best_area = np.inf
    
    # Multi-resolution optimization
    # Stage 1: Coarse grid (fast)
    n_heights_coarse = 1000
    
    def objective_coarse(x):
        return _compute_area(x, n=n, n_heights=n_heights_coarse)
    
    # Evaluate starts
    for pos in starts:
        if time.time() > deadline - 10:
            break
        area = objective_coarse(pos)
        if area < best_area:
            best_area = area
            best_pos = pos.copy()
            
    # Optimize on coarse grid
    current_pos = best_pos.copy()
    try:
        res = minimize(objective_coarse, current_pos, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
        if time.time() < deadline - 10:
            if res.fun < best_area:
                best_area = res.fun
                best_pos = res.x.copy()
    except Exception:
        pass

    # Stage 2: Medium grid
    if time.time() < deadline - 60:
        n_heights_med = 5000
        
        def objective_med(x):
            return _compute_area(x, n=n, n_heights=n_heights_med)
        
        current_pos = best_pos.copy()
        try:
            res = minimize(objective_med, current_pos, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000})
            if time.time() < deadline - 30:
                if res.fun < best_area:
                    best_area = res.fun
                    best_pos = res.x.copy()
        except Exception:
            pass

    # Stage 3: Fine grid (Full resolution)
    if time.time() < deadline - 30:
        n_heights_fine = 20000
        
        def objective_fine(x):
            return _compute_area(x, n=n, n_heights=n_heights_fine)
        
        current_pos = best_pos.copy()
        try:
            res = minimize(objective_fine, current_pos, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000})
            if time.time() < deadline - 10:
                if res.fun < best_area:
                    best_area = res.fun
                    best_pos = res.x.copy()
        except Exception:
            pass

    # Stage 4: Subgradient polish on fine grid
    # This can help if L-BFGS-B got stuck due to non-smoothness
    if time.time() < deadline - 10:
        current_pos = best_pos.copy()
        momentum = np.zeros(n)
        beta = 0.9
        lr = 0.01
        n_iter = 500
        
        for i in range(n_iter):
            if time.time() > deadline - 5:
                break
                
            grad = _subgradient(current_pos, n=n, n_heights=20000)
            momentum = beta * momentum - lr * grad
            current_pos = current_pos + momentum
            
            area = _compute_area(current_pos, n=n, n_heights=20000)
            
            if area < best_area:
                best_area = area
                best_pos = current_pos.copy()
                
            lr *= 0.999

    if not np.all(np.isfinite(best_pos)):
        return np.zeros(n, dtype=np.float64)
        
    return best_pos.astype(np.float64)
