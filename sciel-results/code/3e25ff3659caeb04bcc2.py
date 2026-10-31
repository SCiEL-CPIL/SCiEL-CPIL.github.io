"""Circle packing in unit square – 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

def _is_valid(arr, tol=1e-9):
    """Check that all circles are inside [0,1]^2 with disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _is_valid(arr):
        return arr[:, 2].sum()
    return -np.inf

def _make_grid(n):
    """Create a grid of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _hex_grid(n):
    """Create a hexagonal packing of n circles inside the unit square."""
    # Start with a reasonable hex grid
    side = int(np.ceil(np.sqrt(n)))
    # Hexagonal spacing
    dx = 1.0 / side
    dy = dx * np.sqrt(3) / 2
    cells = []
    for row in range(side + 1):
        offset = (dx / 2) if (row % 2 == 1) else 0
        for col in range(side + 1):
            cx = offset + col * dx
            cy = row * dy
            if 0 <= cx <= 1 and 0 <= cy <= 1:
                r = min(cx, 1 - cx, cy, 1 - cy, dx / 2)
                if r > 0:
                    cells.append([cx, cy, r])
    # Sort by radius descending and take top n
    cells.sort(key=lambda c: -c[2])
    return np.array(cells[:n])

def _optimize_from_start(start, deadline):
    """Local optimization from a starting configuration."""
    x0 = start.flatten()
    
    def objective(v):
        if time.time() > deadline:
            return 1e10
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Penalty for going outside bounds
        pen = 0.0
        for i in range(N):
            pen += max(0, r[i] - x[i]) ** 2
            pen += max(0, r[i] - (1 - x[i])) ** 2
            pen += max(0, r[i] - y[i]) ** 2
            pen += max(0, r[i] - (1 - y[i])) ** 2
            pen += max(0, -r[i]) ** 2
        
        # Penalty for overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        
        # Objective: maximize sum of radii = minimize negative sum + penalty
        return -np.sum(r) + 100 * pen
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12})
    
    if time.time() > deadline:
        return start
    
    x = result.x[:N]
    y = result.x[N:2*N]
    r = result.x[2*N:3*N]
    arr = np.column_stack([x, y, r])
    
    # Clamp to valid region
    for i in range(N):
        r[i] = max(1e-6, min(r[i], min(x[i], 1-x[i], y[i], 1-y[i])))
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple initial configurations
    best_arr = None
    best_score = -np.inf
    
    # Initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_hex_grid(N))
    
    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.1)
        configs.append(np.column_stack([x, y, r]))
    
    for cfg in configs:
        if time.time() > deadline:
            break
        # Try local optimization from this config
        result = _optimize_from_start(cfg, deadline)
        if result is not None:
            s = _score(result)
            if s > best_score:
                best_score = s
                best_arr = result.copy()
    
    # Try differential evolution if we have time
    if time.time() < deadline - 30 and best_arr is not None:
        try:
            def de_objective(v):
                if time.time() > deadline:
                    return 1e10
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:3*N]
                
                pen = 0.0
                for i in range(N):
                    pen += max(0, r[i] - x[i]) ** 2
                    pen += max(0, r[i] - (1 - x[i])) ** 2
                    pen += max(0, r[i] - y[i]) ** 2
                    pen += max(0, r[i] - (1 - y[i])) ** 2
                    pen += max(0, -r[i]) ** 2
                
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            pen += (min_dist - dist) ** 2
                
                return -np.sum(r) + 100 * pen
            
            bounds = [(0, 1)] * (2 * N) + [(0.01, 0.3)] * N
            de_result = differential_evolution(de_objective, bounds, maxiter=50, 
                                               seed=42, tol=1e-6, polish=False)
            
            if de_result.x is not None:
                x = de_result.x[:N]
                y = de_result.x[N:2*N]
                r = de_result.x[2*N:3*N]
                arr = np.column_stack([x, y, r])
                
                for i in range(N):
                    r[i] = max(1e-6, min(r[i], min(x[i], 1-x[i], y[i], 1-y[i])))
                    x[i] = np.clip(x[i], r[i], 1 - r[i])
                    y[i] = np.clip(y[i], r[i], 1 - r[i])
                
                s = _score(arr)
                if s > best_score:
                    best_score = s
                    best_arr = arr.copy()
        except Exception:
            pass
    
    # Final polish if we have time
    if time.time() < deadline - 5 and best_arr is not None:
        polished = _optimize_from_start(best_arr, deadline)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_arr = polished
    
    if best_arr is None:
        # Fallback to grid
        best_arr = _make_grid(N)
    
    return best_arr
