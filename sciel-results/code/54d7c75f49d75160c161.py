"""Circle packing in unit square n=32."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
DEADLINE = 115.0  # safety margin before 120s limit


def _compute_radii(x, y):
    """Compute the maximum radii for given centers."""
    n = len(x)
    # Distance to boundaries
    dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    
    # Set diagonal to inf
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Minimum distance to any other circle
    min_dist = np.min(dist_matrix, axis=1)
    
    # Radius is limited by boundary and half the min distance to other circles
    radii = np.minimum(dist_boundary, min_dist / 2.0)
    radii = np.maximum(radii, 0)
    
    return radii


def _objective(params):
    """Objective to minimize (negative sum of radii)."""
    x = params[0::2]
    y = params[1::2]
    radii = _compute_radii(x, y)
    return -np.sum(radii)


def _push_apart_polish(x, y, max_iters=500, tol=1e-12):
    """
    Polish by pushing apart overlapping circles while preserving sum of radii.
    This moves centers to reduce overlaps without shrinking radii.
    """
    n = len(x)
    x = x.copy()
    y = y.copy()
    radii = _compute_radii(x, y)
    
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist - tol and dist > 1e-15:
                    # Push apart
                    overlap = min_dist - dist
                    # Move each circle half the overlap in opposite directions
                    factor = overlap / (2 * dist)
                    x[i] += factor * dx
                    y[i] += factor * dy
                    x[j] -= factor * dx
                    y[j] -= factor * dy
                    moved = True
        if not moved:
            break
    
    # Clamp to boundaries
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Recompute radii after polishing
    radii = _compute_radii(x, y)
    
    return x, y, radii


def _powell_optimize(x0, y0, maxiter=2000, xtol=1e-10, ftol=1e-10):
    """Run Powell optimization from given initial positions."""
    params0 = np.zeros(2 * N)
    params0[0::2] = x0
    params0[1::2] = y0
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(
            _objective,
            params0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': maxiter, 'xtol': xtol, 'ftol': ftol}
        )
        return res.x
    except Exception:
        return params0


def _generate_grid_init(side, jitter=0.0):
    """Generate a grid-based initial configuration."""
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx] = (i + 0.5) / side
            y0[idx] = (j + 0.5) / side
            if jitter > 0:
                x0[idx] += np.random.randn() * jitter
                y0[idx] += np.random.randn() * jitter
            x0[idx] = np.clip(x0[idx], 0.02, 0.98)
            y0[idx] = np.clip(y0[idx], 0.02, 0.98)
            idx += 1
    if idx < N:
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5
    return x0, y0


def _generate_hex_init(jitter=0.0):
    """Generate a hexagonal grid initial configuration."""
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    idx = 0
    rows = 6
    cols = 6
    dx = 1.0 / cols
    dy = 1.0 / rows
    offset = dx / 2
    
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            x0[idx] = (j + 0.5) * dx + (offset if i % 2 == 1 else 0)
            y0[idx] = (i + 0.5) * dy
            if jitter > 0:
                x0[idx] += np.random.randn() * jitter
                y0[idx] += np.random.randn() * jitter
            x0[idx] = np.clip(x0[idx], 0.02, 0.98)
            y0[idx] = np.clip(y0[idx], 0.02, 0.98)
            idx += 1
    if idx < N:
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5
    return x0, y0


def _generate_asymmetric_grid_init(jitter=0.0):
    """Generate an asymmetric grid initial configuration."""
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    idx = 0
    # Use different spacings in x and y
    nx = 5
    ny = 7
    dx = 1.0 / nx
    dy = 1.0 / ny
    
    for i in range(ny):
        for j in range(nx):
            if idx >= N:
                break
            x0[idx] = (j + 0.5) * dx
            y0[idx] = (i + 0.5) * dy
            if jitter > 0:
                x0[idx] += np.random.randn() * jitter
                y0[idx] += np.random.randn() * jitter
            x0[idx] = np.clip(x0[idx], 0.02, 0.98)
            y0[idx] = np.clip(y0[idx], 0.02, 0.98)
            idx += 1
    if idx < N:
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5
    return x0, y0


def _generate_random_init(scale=0.1):
    """Generate a random initial configuration near grid points."""
    x0, y0 = _generate_grid_init(6, jitter=0)
    x0 += np.random.randn(N) * scale
    y0 += np.random.randn(N) * scale
    x0 = np.clip(x0, 0.02, 0.98)
    y0 = np.clip(y0, 0.02, 0.98)
    return x0, y0


def construct():
    start_time = time.time()
    
    best_params = None
    best_obj = np.inf
    
    def update_best(params):
        nonlocal best_params, best_obj
        obj = _objective(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
    
    # Phase 1: Multiple diverse initializations with Powell
    np.random.seed(42)
    
    initializations = []
    
    # Grid initialization
    x_grid, y_grid = _generate_grid_init(6)
    initializations.append(('grid', x_grid, y_grid))
    
    # Hexagonal initialization
    x_hex, y_hex = _generate_hex_init()
    initializations.append(('hex', x_hex, y_hex))
    
    # Asymmetric grid
    x_asym, y_asym = _generate_asymmetric_grid_init()
    initializations.append(('asym', x_asym, y_asym))
    
    # Random perturbations of grid
    for _ in range(3):
        x_rand, y_rand = _generate_random_init(scale=0.05)
        initializations.append(('rand', x_rand, y_rand))
    
    # Run Powell from each initialization
    for name, x0, y0 in initializations:
        if time.time() > start_time + DEADLINE - 30:
            break
        
        params = _powell_optimize(x0, y0, maxiter=3000)
        update_best(params)
        
        # Push-apart polish
        x_p = params[0::2]
        y_p = params[1::2]
        x_p, y_p, _ = _push_apart_polish(x_p, y_p, max_iters=200)
        params_polished = np.zeros(2 * N)
        params_polished[0::2] = x_p
        params_polished[1::2] = y_p
        update_best(params_polished)
    
    # Phase 2: Local refinement with small perturbations
    if best_params is not None and time.time() < start_time + DEADLINE - 10:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        radii_best = _compute_radii(x_best, y_best)
        sum_best = np.sum(radii_best)
        
        # Try moving each circle in small steps
        for _ in range(100):
            if time.time() > start_time + DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                for dx_try in [-0.005, -0.002, 0.0, 0.002, 0.005]:
                    for dy_try in [-0.005, -0.002, 0.0, 0.002, 0.005]:
                        x_try = x_best.copy()
                        y_try = y_best.copy()
                        x_try[i] += dx_try
                        y_try[i] += dy_try
                        
                        if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                            continue
                        
                        radii_try = _compute_radii(x_try, y_try)
                        sum_try = np.sum(radii_try)
                        
                        if sum_try > sum_best + 1e-12:
                            x_best = x_try
                            y_best = y_try
                            radii_best = radii_try
                            sum_best = sum_try
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        # Update best_params
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = _objective(best_params)
        
        # Final push-apart polish
        x_p = best_params[0::2]
        y_p = best_params[1::2]
        x_p, y_p, _ = _push_apart_polish(x_p, y_p, max_iters=500)
        params_polished = np.zeros(2 * N)
        params_polished[0::2] = x_p
        params_polished[1::2] = y_p
        update_best(params_polished)
    
    # Final construction
    if best_params is None:
        # Fallback to grid
        x_grid, y_grid = _generate_grid_init(6)
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_grid
        best_params[1::2] = y_grid
    
    x = best_params[0::2]
    y = best_params[1::2]
    radii = _compute_radii(x, y)
    
    # Shrink radii slightly to ensure strict disjointness (floating point safety)
    radii *= 0.99999
    
    result = np.column_stack((x, y, radii))
    return result
