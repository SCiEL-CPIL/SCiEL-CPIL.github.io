"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize
import itertools

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Pack 21 circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    
    Strategy:
    1. Start with a grid-like initialization.
    2. Use a smooth optimization approach to maximize sum of radii.
    3. The rectangle dimensions (w, h) are also variables with constraint 2*(w+h) <= 4.
    4. Circles must be inside the rectangle and non-overlapping.
    
    We'll use a multi-start local optimization with smooth constraints.
    """
    if _time_left() < 5.0:
        # Fallback: simple grid
        return _simple_grid()
    
    best_result = None
    best_sum = -1.0
    
    # Multiple initializations
    n_starts = 8
    
    for start_idx in range(n_starts):
        if _time_left() < 10.0:
            break
        
        # Initialize positions and radii
        x0 = _initialize(start_idx)
        
        # Optimize
        result = _optimize(x0)
        
        if result is not None:
            sum_radii = np.sum(result[:, 2])
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result.copy()
    
    if best_result is not None:
        return best_result
    else:
        return _simple_grid()


def _simple_grid():
    """Simple fallback: 5x5 grid, use first 21 cells."""
    side = 5
    # Rectangle: let's use a square of side 1 (perimeter 4)
    # 5x5 grid in [0,1]x[0,1]
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _initialize(start_idx):
    """
    Initialize N circles with positions and radii.
    Returns array of shape (N, 3) with [x, y, r].
    """
    # Try different rectangle aspect ratios
    aspect_ratios = [1.0, 1.5, 0.66, 1.2, 0.8, 2.0, 0.5, 1.3]
    ar = aspect_ratios[start_idx % len(aspect_ratios)]
    
    # Perimeter = 4, so w + h = 2
    h = 2.0 / (1.0 + ar)
    w = 2.0 - h
    
    # Grid initialization
    n_cols = max(1, int(np.round(np.sqrt(N * ar))))
    n_rows = max(1, int(np.ceil(N / n_cols)))
    
    # Adjust to fit exactly N circles
    cells = []
    count = 0
    for i in range(n_cols):
        for j in range(n_rows):
            if count >= N:
                break
            cx = (i + 0.5) * w / n_cols
            cy = (j + 0.5) * h / n_rows
            r = min(w / n_cols, h / n_rows) / 2.0 * 0.95  # 5% margin
            cells.append([cx, cy, r])
            count += 1
    
    # Add remaining circles if needed
    while len(cells) < N:
        cx = np.random.uniform(0, w)
        cy = np.random.uniform(0, h)
        r = 0.02
        cells.append([cx, cy, r])
    
    return np.array(cells[:N])


def _optimize(x0):
    """
    Optimize the circle packing using gradient-based optimization.
    
    Variables: [x_1, y_1, r_1, ..., x_N, y_N, r_N, w, h]
    Total: 3*N + 2 = 65 variables
    
    Objective: maximize sum(r_i)
    Constraints:
    1. 2*(w + h) <= 4  (perimeter)
    2. 0 <= x_i <= w - r_i  (inside rectangle horizontally)
    3. 0 <= y_i <= h - r_i  (inside rectangle vertically)
    4. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  (non-overlap)
    5. r_i > 0
    """
    n = len(x0)
    
    # Variables: x (n), y (n), r (n), w, h
    # Flatten
    x = x0[:, 0].copy()
    y = x0[:, 1].copy()
    r = x0[:, 2].copy()
    w = max(0.1, np.max(x) + np.max(r) + 0.1)
    h = max(0.1, np.max(y) + np.max(r) + 0.1)
    
    # Normalize to fit perimeter constraint
    perim = 2 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim
        x *= scale
        y *= scale
        r *= scale
        w *= scale
        h *= scale
    
    def objective(vars):
        """Negative sum of radii (minimize negative)."""
        xi = vars[:n]
        yi = vars[n:2*n]
        ri = vars[2*n:3*n]
        wv = vars[3*n]
        hv = vars[3*n+1]
        
        # Penalty for perimeter violation
        perim = 2 * (wv + hv)
        if perim > 4.0:
            penalty = 100.0 * (perim - 4.0)
        else:
            penalty = 0.0
        
        # Penalty for circles outside rectangle
        for i in range(n):
            if xi[i] < ri[i] or xi[i] > wv - ri[i]:
                penalty += 100.0 * max(0, ri[i] - xi[i]) + 100.0 * max(0, xi[i] - (wv - ri[i]))
            if yi[i] < ri[i] or yi[i] > hv - ri[i]:
                penalty += 100.0 * max(0, ri[i] - yi[i]) + 100.0 * max(0, yi[i] - (hv - ri[i]))
        
        # Penalty for overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-10)
                min_dist = ri[i] + ri[j]
                if dist < min_dist:
                    penalty += 100.0 * (min_dist - dist)
        
        # Penalty for negative radius
        for i in range(n):
            if ri[i] < 0:
                penalty += 100.0 * (-ri[i])
        
        return -np.sum(ri) + penalty
    
    # Initial variable vector
    vars0 = np.concatenate([x, y, r, [w, h]])
    
    # Bounds
    # x, y: [0, 2] (max dimension is 2)
    # r: [0, 1]
    # w, h: [0.01, 2]
    bounds = [(0, 2.0)] * n + [(0, 2.0)] * n + [(0.001, 1.0)] * n + [(0.01, 2.0), (0.01, 2.0)]
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(objective, vars0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if _time_left() < 5.0:
            return None
        
        # Extract solution
        xi = result.x[:n]
        yi = result.x[n:2*n]
        ri = result.x[2*n:3*n]
        wv = result.x[3*n]
        hv = result.x[3*n+1]
        
        # Check validity
        perim = 2 * (wv + hv)
        if perim > 4.0 + 1e-6:
            return None
        
        valid = True
        for i in range(n):
            if ri[i] <= 0:
                valid = False
                break
            if xi[i] < ri[i] - 1e-6 or xi[i] > wv - ri[i] + 1e-6:
                valid = False
                break
            if yi[i] < ri[i] - 1e-6 or yi[i] > hv - ri[i] + 1e-6:
                valid = False
                break
        
        if not valid:
            return None
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < ri[i] + ri[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        return np.column_stack([xi, yi, ri])
    
    except Exception:
        return None


# EVOLVE-BLOCK-END
