"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _valid(arr):
    """Check if the packing is valid: circles inside [0,1]^2 and disjoint interiors."""
    if arr is None or len(arr) != N:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Check inside unit square
    if np.any(r < -1e-9) or np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7) or np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    # Check disjoint interiors: distance between centers >= sum of radii
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _score(arr):
    """Compute the objective: sum of radii. Return -inf if invalid."""
    if not _valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])

def _init_grid():
    """Initialize with a grid of equal circles."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells[:N])

def _penalty(arr, w=100.0):
    """
    Penalty function for optimization.
    arr: (N, 3) array with x, y, r
    Returns a scalar penalty to minimize (negative objective + constraints).
    """
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Objective: maximize sum of r => minimize -sum(r)
    obj = -np.sum(r)
    
    # Constraint: circles inside unit square
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # r >= 0
    pen = 0.0
    pen += w * np.sum(np.maximum(0, r - x))          # x - r >= 0
    pen += w * np.sum(np.maximum(0, x + r - 1))       # x + r <= 1
    pen += w * np.sum(np.maximum(0, r - y))           # y - r >= 0
    pen += w * np.sum(np.maximum(0, y + r - 1))       # y + r <= 1
    pen += w * np.sum(np.maximum(0, -r))              # r >= 0
    
    # Constraint: disjoint interiors
    # dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            violation = (r[i] + r[j]) - dist
            pen += w * max(0.0, violation) ** 2
    
    return obj + pen

def _optimize_from(start, max_iter=1000, tol=1e-8):
    """
    Run gradient-based optimization from a starting point.
    Uses L-BFGS-B with bounds.
    """
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # Reshape to 1D for scipy
    x0 = start.flatten()
    
    def fun_flat(v):
        arr = v.reshape(N, 3)
        return _penalty(arr)
    
    res = minimize(fun_flat, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    
    best_arr = res.x.reshape(N, 3)
    
    # Ensure validity by clamping and fixing small violations
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 0, 0.5)
    
    # Fix boundary violations
    for i in range(N):
        x, y, r = best_arr[i]
        r = min(r, x, 1 - x, y, 1 - y)
        best_arr[i, 2] = max(0, r)
    
    # Fix overlap violations by reducing radii
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = best_arr[i, 0] - best_arr[j, 0]
                dy = best_arr[i, 1] - best_arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < best_arr[i, 2] + best_arr[j, 2] - 1e-9:
                    # Reduce the smaller radius
                    if best_arr[i, 2] < best_arr[j, 2]:
                        best_arr[i, 2] = max(0, dist - best_arr[j, 2])
                    else:
                        best_arr[j, 2] = max(0, dist - best_arr[i, 2])
                    changed = True
        if not changed:
            break
    
    return best_arr

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with grid initialization
    current = _init_grid()
    best = current.copy()
    best_score = _score(best)
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Generate random initialization
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.15, N)
        
        start = np.column_stack([x, y, r])
        
        # Optimize
        result = _optimize_from(start, max_iter=500, tol=1e-7)
        
        result_score = _score(result)
        if result_score > best_score:
            best = result
            best_score = result_score
    
    # Final polish on the best solution
    if time.time() < deadline - 5:
        polished = _optimize_from(best, max_iter=2000, tol=1e-10)
        polished_score = _score(polished)
        if polished_score > best_score:
            best = polished
            best_score = polished_score
    
    # Ensure final validity
    if not _valid(best):
        # Fallback to grid
        best = _init_grid()
    
    return best
# EVOLVE-BLOCK-END
