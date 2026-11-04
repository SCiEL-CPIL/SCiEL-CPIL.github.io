"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after ~145s to stay under 150s wall-clock
_DEADLINE = 145.0
_START_TIME = time.time()

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _validity_check(arr):
    """Check that circles are disjoint and inside a perimeter-4 rectangle."""
    if len(arr) != N:
        return False
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    # All radii must be positive
    if np.any(rs <= 0):
        return False
    # Bounding box
    x_min, x_max = xs.min() - rs, xs.max() + rs
    y_min, y_max = ys.min() - rs, ys.max() + rs
    W = x_max - x_min
    H = y_max - y_min
    if W <= 0 or H <= 0:
        return False
    # Perimeter constraint: 2*(W+H) <= 4
    if 2.0 * (W + H) > 4.0 + 1e-6:
        return False
    # Disjoint interiors: for all i != j, dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - 1e-7:
                return False
    return True

def _objective_sum_radii(arr):
    """Return sum of radii (higher is better). We minimize negative."""
    return -np.sum(arr[:, 2])

def _hex_init(W, H, n=21):
    """Generate a hexagonal packing initialization inside W x H rectangle."""
    # Try to fit hexagonal grid
    # Hex spacing: dx = r*sqrt(3), dy = r*1.5 for rows
    # Start with a guess for radius
    # For n circles in a hex grid: roughly rows * cols = n
    # Let's try various aspect ratios for the grid
    best_init = None
    best_score = -np.inf
    
    # Try different row/col combinations
    for rows in range(3, 8):
        cols = int(np.ceil(n / rows))
        # Hex grid: alternate rows offset
        # Estimate radius from W and H
        # Horizontal: cols * sqrt(3) * r <= W  => r <= W / (cols * sqrt(3))
        # Vertical: (rows - 1) * 1.5 * r + 2*r <= H => r <= H / (1.5*(rows-1) + 2)
        r_x = W / (cols * np.sqrt(3))
        r_y = H / (1.5 * (rows - 1) + 2.0)
        r = min(r_x, r_y) * 0.9  # margin
        if r <= 0:
            continue
        
        points = []
        for row in range(rows):
            for col in range(cols):
                if len(points) >= n:
                    break
                x = r + col * np.sqrt(3) * r + (0.5 * np.sqrt(3) * r if row % 2 else 0)
                y = r + row * 1.5 * r
                if x + r <= W and y + r <= H:
                    points.append([x, y, r])
        
        if len(points) < n:
            continue
        
        points = points[:n]
        arr = np.array(points)
        # Center in rectangle
        arr[:, 0] -= arr[:, 0].min()
        arr[:, 1] -= arr[:, 1].min()
        
        # Check validity and score
        if _validity_check(arr):
            score = np.sum(arr[:, 2])
            if score > best_score:
                best_score = score
                best_init = arr.copy()
    
    if best_init is not None:
        return best_init
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:n])

def _optimize_aspect(W, H, init=None):
    """Optimize circle positions and radii for given rectangle W x H (perimeter 2*(W+H)=4)."""
    if init is None:
        init = _hex_init(W, H)
    
    x0 = init.flatten().copy()
    
    # Objective: maximize sum of radii subject to constraints
    # We'll use a penalty-based approach with L-BFGS-B for speed
    # Variables: x_i, y_i, r_i for i=0..N-1
    
    def _penalty_objective(x):
        arr = x.reshape(N, 3)
        xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
        
        # Penalty for going outside rectangle
        pen = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                pen += (rs[i] - xs[i])**2
            if xs[i] > W - rs[i]:
                pen += (xs[i] - (W - rs[i]))**2
            if ys[i] < rs[i]:
                pen += (rs[i] - ys[i])**2
            if ys[i] > H - rs[i]:
                pen += (ys[i] - (H - rs[i]))**2
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    pen += overlap**2
        
        # Penalty for negative radii
        neg_r = np.clip(-rs, 0, None)
        pen += np.sum(neg_r**2)
        
        return -np.sum(rs) + 100.0 * pen
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, W))  # x
        bounds.append((0, H))  # y
        bounds.append((1e-6, min(W, H) / 2))  # r
    
    # Optimize
    result = minimize(_penalty_objective, x0, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    arr = result.x.reshape(N, 3)
    
    # Clamp to valid region
    for i in range(N):
        arr[i, 0] = np.clip(arr[i, 0], arr[i, 2], W - arr[i, 2])
        arr[i, 1] = np.clip(arr[i, 1], arr[i, 2], H - arr[i, 2])
        arr[i, 2] = max(arr[i, 2], 1e-6)
    
    # Check validity; if invalid, try to fix by reducing radii slightly
    if not _validity_check(arr):
        # Try reducing all radii by a small factor
        for factor in [0.99, 0.98, 0.95, 0.9]:
            arr_test = arr.copy()
            arr_test[:, 2] *= factor
            if _validity_check(arr_test):
                arr = arr_test
                break
    
    return arr, np.sum(arr[:, 2])

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_arr = None
    best_score = 0.0
    
    # Explore various aspect ratios
    # Perimeter 2*(W+H) = 4 => W+H = 2
    aspect_ratios = []
    # W from 0.2 to 1.8 (since W+H=2, H=2-W)
    for W in np.linspace(0.2, 1.8, 20):
        H = 2.0 - W
        if H > 0:
            aspect_ratios.append((W, H))
    
    # Sort by how close to square (square tends to pack better for moderate n)
    aspect_ratios.sort(key=lambda wh: abs(wh[0] - wh[1]))
    
    for W, H in aspect_ratios:
        if _remaining_time() < 10.0:
            break
        
        init = _hex_init(W, H)
        arr, score = _optimize_aspect(W, H, init)
        
        if _validity_check(arr) and score > best_score:
            best_score = score
            best_arr = arr.copy()
        
        # Also try with slightly perturbed initialization
        if _remaining_time() < 15.0:
            break
        
        np.random.seed(42)
        init2 = init + np.random.normal(0, 0.01, init.shape)
        # Clamp init2 to valid bounds
        for i in range(N):
            init2[i, 0] = np.clip(init2[i, 0], init2[i, 2], W - init2[i, 2])
            init2[i, 1] = np.clip(init2[i, 1], init2[i, 2], H - init2[i, 2])
        
        arr2, score2 = _optimize_aspect(W, H, init2)
        if _validity_check(arr2) and score2 > best_score:
            best_score = score2
            best_arr = arr2.copy()
    
    # Final polish: if we have a good solution, try a few more iterations
    if best_arr is not None and _remaining_time() > 20.0:
        # Find the best aspect ratio from our best solution
        xs, ys, rs = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
        x_min, x_max = xs.min() - rs, xs.max() + rs
        y_min, y_max = ys.min() - rs, ys.max() + rs
        W_best = x_max - x_min
        H_best = y_max - y_min
        
        # Refine with more iterations
        x0 = best_arr.flatten().copy()
        
        def _penalty_objective_refine(x):
            arr = x.reshape(N, 3)
            xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
            
            pen = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    pen += (rs[i] - xs[i])**2
                if xs[i] > W_best - rs[i]:
                    pen += (xs[i] - (W_best - rs[i]))**2
                if ys[i] < rs[i]:
                    pen += (rs[i] - ys[i])**2
                if ys[i] > H_best - rs[i]:
                    pen += (ys[i] - (H_best - rs[i]))**2
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        pen += overlap**2
            
            neg_r = np.clip(-rs, 0, None)
            pen += np.sum(neg_r**2)
            
            return -np.sum(rs) + 100.0 * pen
        
        bounds = []
        for i in range(N):
            bounds.append((0, W_best))
            bounds.append((0, H_best))
            bounds.append((1e-6, min(W_best, H_best) / 2))
        
        result = minimize(_penalty_objective_refine, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        arr_refined = result.x.reshape(N, 3)
        for i in range(N):
            arr_refined[i, 0] = np.clip(arr_refined[i, 0], arr_refined[i, 2], W_best - arr_refined[i, 2])
            arr_refined[i, 1] = np.clip(arr_refined[i, 1], arr_refined[i, 2], H_best - arr_refined[i, 2])
            arr_refined[i, 2] = max(arr_refined[i, 2], 1e-6)
        
        if _validity_check(arr_refined):
            score_refined = np.sum(arr_refined[:, 2])
            if score_refined > best_score:
                best_arr = arr_refined
                best_score = score_refined
    
    if best_arr is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr
# EVOLVE-BLOCK-END
