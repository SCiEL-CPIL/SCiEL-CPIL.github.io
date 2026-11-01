"""Circle packing in unit square n=32."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 32
DEADLINE = 110.0  # safety margin before 120s limit


def _objective(params):
    """
    params: shape (3N,) -> [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Objective: maximize sum of radii subject to:
      1. Circles inside [0,1]^2: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      2. Disjoint interiors: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    We minimize the negative sum of radii plus penalty for violations.
    """
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for being outside the square
    penalty = 0.0
    for i in range(n):
        # Must be inside [0,1]^2
        if r[i] > x[i]:
            penalty += (r[i] - x[i]) ** 2
        if r[i] > 1.0 - x[i]:
            penalty += (r[i] - (1.0 - x[i])) ** 2
        if r[i] > y[i]:
            penalty += (r[i] - y[i]) ** 2
        if r[i] > 1.0 - y[i]:
            penalty += (r[i] - (1.0 - y[i])) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return -np.sum(r) + 1000.0 * penalty


def _is_valid(params, tol=1e-6):
    """Check if the configuration is valid (all circles inside square, no overlaps)."""
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    for i in range(n):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1.0 + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _grid_init(side_x, side_y, n):
    """Create a grid-based initialization."""
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            x = (i + 0.5) / side_x
            y = (j + 0.5) / side_y
            r = 0.5 / max(side_x, side_y) * 0.95
            cells.append([x, y, r])
    
    # Shuffle and take first n
    np.random.shuffle(cells)
    cells = cells[:n]
    
    params = np.array([v for c in cells for v in c])
    return params


def _random_init(n):
    """Create a random initialization."""
    params = np.zeros(3 * n)
    for i in range(n):
        params[3 * i] = np.random.uniform(0.1, 0.9)      # x
        params[3 * i + 1] = np.random.uniform(0.1, 0.9)  # y
        params[3 * i + 2] = np.random.uniform(0.02, 0.15) # r
    return params


def _perturb(params, scale=0.05):
    """Add random perturbation to parameters."""
    p = params.copy()
    p[0::3] += np.random.normal(0, scale, size=len(p[0::3]))
    p[1::3] += np.random.normal(0, scale, size=len(p[1::3]))
    p[2::3] += np.random.normal(0, scale * 0.5, size=len(p[2::3]))
    # Clip to valid ranges
    p[0::3] = np.clip(p[0::3], 0.01, 0.99)
    p[1::3] = np.clip(p[1::3], 0.01, 0.99)
    p[2::3] = np.clip(p[2::3], 0.005, 0.4)
    return p


def construct():
    start_time = time.time()
    best_params = None
    best_score = -np.inf
    
    # Define bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))   # x
        bounds.append((0.01, 0.99))   # y
        bounds.append((0.005, 0.4))   # r
    
    # Strategy 1: Differential evolution (global search)
    if time.time() - start_time < DEADLINE * 0.4:
        try:
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                update_interval=1
            )
            if _is_valid(de_result.x):
                score = np.sum(de_result.x[2::3])
                if score > best_score:
                    best_score = score
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple local optimizations from diverse starts
    if time.time() - start_time < DEADLINE * 0.8:
        # Grid-based inits with different aspect ratios
        grid_configs = [
            (4, 8), (8, 4), (5, 7), (7, 5), (6, 6), (4, 6), (6, 4)
        ]
        
        for side_x, side_y in grid_configs:
            if time.time() - start_time > DEADLINE * 0.75:
                break
            
            init = _grid_init(side_x, side_y, N)
            
            # Try multiple optimizers
            for method in ['Nelder-Mead', 'Powell']:
                if time.time() - start_time > DEADLINE * 0.75:
                    break
                
                try:
                    result = minimize(
                        _objective, init,
                        method=method,
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if _is_valid(result.x):
                        score = np.sum(result.x[2::3])
                        if score > best_score:
                            best_score = score
                            best_params = result.x.copy()
                except Exception:
                    pass
        
        # Random inits
        for trial in range(20):
            if time.time() - start_time > DEADLINE * 0.75:
                break
            
            init = _random_init(N)
            # Add perturbation at multiple scales
            for scale in [0.1, 0.05, 0.02]:
                if time.time() - start_time > DEADLINE * 0.75:
                    break
                perturbed = _perturb(init, scale)
                try:
                    result = minimize(
                        _objective, perturbed,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if _is_valid(result.x):
                        score = np.sum(result.x[2::3])
                        if score > best_score:
                            best_score = score
                            best_params = result.x.copy()
                except Exception:
                    pass
    
    # Final refinement if we have a good solution
    if best_params is not None and time.time() - start_time < DEADLINE * 0.9:
        try:
            final_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if _is_valid(final_result.x):
                score = np.sum(final_result.x[2::3])
                if score > best_score:
                    best_score = score
                    best_params = final_result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = best_params[0::3]
    result[:, 1] = best_params[1::3]
    result[:, 2] = best_params[2::3]
    
    # Ensure validity one more time
    if not _is_valid(best_params):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return result
