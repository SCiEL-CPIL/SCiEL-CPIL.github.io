"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 140.0  # safety margin before 150s kill

def _score(x_flat):
    """Compute the radius score for a flat array of 2N coordinates."""
    x = x_flat.reshape(N, 2)
    # Check bounds
    if np.any(np.abs(x) > 0.5 + 1e-10):
        return 0.0
    # Pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(x))
    r = min(min_pair / 2.0, min_boundary)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return r

def _soft_min(a, beta=20.0):
    """Soft minimum of array a using log-sum-exp style."""
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    m = np.min(a)
    return m + np.log(np.sum(np.exp(beta * (a - m)))) / beta

def _objective(x_flat):
    """Objective to minimize: -radius (so maximizing radius)."""
    x = x_flat.reshape(N, 2)
    # Penalty for going outside
    bound_viol = np.maximum(0.0, np.abs(x) - 0.5)
    penalty = 1000.0 * np.sum(bound_viol**2)
    
    # Pairwise distances (soft min for smoothness)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = _soft_min(dist, beta=30.0) / 2.0
    
    # Boundary distance (soft min)
    min_boundary = _soft_min(0.5 - np.abs(x), beta=30.0)
    
    r = min(min_pair, min_boundary)
    return -r + penalty

def _hexagonal_init():
    """Generate a hexagonal close-packed initial configuration."""
    # For 200 circles, use a hexagonal grid
    # Hex grid with spacing s: rows offset
    # Try to fit as many as possible
    best_points = None
    best_score = 0.0
    
    for s in np.linspace(0.03, 0.05, 20):
        points = []
        row = 0
        while len(points) < N and row < 30:
            offset = (s * np.sqrt(3) / 2) * (row % 2)
            x_start = -0.5 + s + offset
            x = x_start
            col = 0
            while x <= 0.5 - s and len(points) < N:
                y = -0.5 + s + row * s * np.sqrt(3) / 2
                if y <= 0.5 - s:
                    points.append([x, y])
                x += s
                col += 1
            row += 1
        if len(points) >= N:
            points = np.array(points[:N])
            sc = _score(points.flatten())
            if sc > best_score:
                best_score = sc
                best_points = points
    if best_points is not None:
        return best_points
    # Fallback: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _random_init(n_random=10):
    """Try several random initializations and return the best."""
    best = None
    best_sc = 0.0
    for i in range(n_random):
        pts = np.random.uniform(-0.49, 0.49, size=(N, 2))
        sc = _score(pts.flatten())
        if sc > best_sc:
            best_sc = sc
            best = pts
    return best

def construct():
    start_time = time.time()
    
    # Initial configurations
    init_hex = _hexagonal_init()
    init_rand = _random_init(5)
    
    candidates = [init_hex, init_rand]
    # Remove None
    candidates = [c for c in candidates if c is not None]
    
    best_x = None
    best_score = 0.0
    for c in candidates:
        sc = _score(c.flatten())
        if sc > best_score:
            best_score = sc
            best_x = c.copy()
    
    # Local optimization with L-BFGS-B
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        try:
            res = minimize(
                _objective,
                init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            sc = _score(res.x)
            if sc > best_score:
                best_score = sc
                best_x = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Polish pass with tighter tolerances on the best
    if time.time() - start_time < DEADLINE - 5 and best_x is not None:
        try:
            res2 = minimize(
                _objective,
                best_x.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
            )
            sc2 = _score(res2.x)
            if sc2 > best_score:
                best_score = sc2
                best_x = res2.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Final validity check
    if best_x is None:
        return init_hex
    result = best_x
    # Clamp to be safe
    result = np.clip(result, -0.5, 0.5)
    return result
