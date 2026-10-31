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
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
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
    side = int(np.ceil(np.sqrt(n)))
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
    cells.sort(key=lambda c: -c[2])
    return np.array(cells[:n])

def _objective(v, deadline):
    """
    Vectorized objective function for optimization.
    v: flat array of [x1..xN, y1..yN, r1..rN]
    Returns: -sum(r) + penalty
    """
    if time.time() > deadline:
        return 1e10
    
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Penalty for going outside bounds (vectorized)
    pen = 0.0
    pen += np.sum(np.maximum(0, r - x)**2)
    pen += np.sum(np.maximum(0, r - (1 - x))**2)
    pen += np.sum(np.maximum(0, r - y)**2)
    pen += np.sum(np.maximum(0, r - (1 - y))**2)
    pen += np.sum(np.maximum(0, -r)**2)
    
    # Penalty for overlapping circles (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_dist - dist)
    pen += np.sum(overlap**2)
    
    return -np.sum(r) + 100 * pen

def _postprocess(arr, deadline):
    """
    Post-process the result to ensure validity by shrinking radii if necessary.
    """
    if arr is None:
        return None
    
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    # First, clamp radii to be within bounds
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(1e-6, min(r[i], max_r))
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Iterative shrink to resolve overlaps (vectorized)
    for _ in range(200):
        if time.time() > deadline:
            break
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        req = r[:, None] + r[None, :]
        overlap = np.maximum(0, req - dist)
        
        if np.max(overlap) < 1e-10:
            break
        
        # Shrink radii proportionally to overlap
        for i in range(N):
            for j in range(i + 1, N):
                if overlap[i, j] > 0:
                    shrink = overlap[i, j] / 2 + 1e-7
                    r[i] -= shrink
                    r[j] -= shrink
    
    # Final clamp
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(1e-6, min(r[i], max_r))
    
    return np.column_stack([x, y, r])

def _optimize_from_start(start, deadline):
    """Local optimization from a starting configuration using L-BFGS-B."""
    x0 = start.flatten()
    
    try:
        result = minimize(lambda v: _objective(v, deadline), x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
        
        if time.time() > deadline:
            return start
        
        x = result.x[:N]
        y = result.x[N:2*N]
        r = result.x[2*N:3*N]
        arr = np.column_stack([x, y, r])
        arr = _postprocess(arr, deadline)
        
        if arr is not None:
            return arr
    except Exception:
        pass
    
    return start

def _optimize_nm(start, deadline):
    """Nelder-Mead optimization from a starting configuration."""
    x0 = start.flatten()
    
    try:
        result = minimize(lambda v: _objective(v, deadline), x0, method='Nelder-Mead', 
                          options={'maxiter': 10000, 'maxfev': 50000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() > deadline:
            return start
        
        x = result.x[:N]
        y = result.x[N:2*N]
        r = result.x[2*N:3*N]
        arr = np.column_stack([x, y, r])
        arr = _postprocess(arr, deadline)
        
        if arr is not None:
            return arr
    except Exception:
        pass
    
    return start

def _make_initial_configs(rng):
    """Generate diverse initial configurations."""
    configs = []
    configs.append(_make_grid(N))
    configs.append(_hex_grid(N))
    
    # Random with varied radii
    for trial in range(20):
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.05, 0.15, N)
        configs.append(np.column_stack([x, y, r]))
    
    # Structured random: place circles in a loose grid then jitter
    side = 5
    for trial in range(10):
        x = np.array([(i + 0.5) / side + rng.normal(0, 0.02, 1)[0] for i in range(side) for _ in range(6)][:N])
        y = np.array([(j + 0.5) / side + rng.normal(0, 0.02, 1)[0] for j in range(6) for i in range(side)][:N])
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        r = np.full(N, 0.08 + rng.uniform(0, 0.04))
        configs.append(np.column_stack([x, y, r]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_arr = None
    best_score = -np.inf
    
    configs = _make_initial_configs(rng)
    
    # Phase 1: Quick L-BFGS-B on all configs
    for cfg in configs:
        if time.time() > deadline:
            break
        
        result = _optimize_from_start(cfg, deadline)
        if result is not None:
            s = _score(result)
            if s > best_score:
                best_score = s
                best_arr = result.copy()
    
    # Phase 2: Nelder-Mead on top candidates
    if best_arr is not None and time.time() < deadline - 20:
        # Use the best config and a few random ones
        nm_configs = [best_arr]
        for cfg in configs[:5]:
            nm_configs.append(cfg)
        
        for cfg in nm_configs:
            if time.time() > deadline - 5:
                break
            
            result_nm = _optimize_nm(cfg, deadline)
            if result_nm is not None:
                s = _score(result_nm)
                if s > best_score:
                    best_score = s
                    best_arr = result_nm.copy()
    
    # Phase 3: Differential evolution if time permits
    if time.time() < deadline - 40 and best_arr is not None:
        try:
            bounds = [(0, 1)] * (2 * N) + [(0.02, 0.25)] * N
            de_result = differential_evolution(
                lambda v: _objective(v, deadline), bounds, 
                maxiter=100, seed=42, tol=1e-8, polish=False,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7
            )
            
            if de_result.x is not None:
                x = de_result.x[:N]
                y = de_result.x[N:2*N]
                r = de_result.x[2*N:3*N]
                arr = np.column_stack([x, y, r])
                arr = _postprocess(arr, deadline)
                
                if arr is not None:
                    s = _score(arr)
                    if s > best_score:
                        best_score = s
                        best_arr = arr.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with multiple local optimizations
    if time.time() < deadline - 10 and best_arr is not None:
        # Polish from best
        polished = _optimize_from_start(best_arr, deadline)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_arr = polished
        
        # Try polishing from perturbed best
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = best_arr.copy()
            perturbed[:, 0] += rng.normal(0, 0.01, N)
            perturbed[:, 1] += rng.normal(0, 0.01, N)
            perturbed = np.clip(perturbed, 0, 1)
            
            polished = _optimize_from_start(perturbed, deadline)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_arr = polished
    
    if best_arr is None:
        best_arr = _make_grid(N)
    
    return best_arr
