"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _clamp(x, lo, hi):
    return np.clip(x, lo, hi)

def _penalty(x, y, r):
    """Penalty for constraint violations (positive = bad)."""
    # Boundary violations
    pen = 0.0
    for i in range(N):
        xi, yi, ri = x[i], y[i], r[i]
        # Must be inside [0,1]^2
        pen += max(0, ri - xi) ** 2
        pen += max(0, ri - yi) ** 2
        pen += max(0, ri - (1 - xi)) ** 2
        pen += max(0, ri - (1 - yi)) ** 2
    # Overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            di = x[i] - x[j]
            dj = y[i] - y[j]
            dist = np.sqrt(di * di + dj * dj)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return pen

def _objective(params):
    """Objective to minimize: -sum(r) + penalty."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]
    return -np.sum(r) + 100.0 * _penalty(x, y, r)

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol or r[i] < -tol:
            return False
        if x[i] + r[i] > 1 + tol or y[i] + r[i] > 1 + tol:
            return False
        if x[i] - r[i] < -tol or y[i] - r[i] < -tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _score(x, y, r):
    """Score is sum of radii if valid, else 0."""
    if _is_valid(x, y, r):
        return np.sum(r)
    return 0.0

def _grid_init():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01))
    return np.array(cells[:N])

def _hex_init():
    """Hexagonal packing initialization."""
    # Use a hexagonal lattice pattern
    rows = int(np.ceil(N / 4))  # approximately
    pts = []
    r = 0.08
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    for i in range(rows + 2):
        for j in range(8):
            if len(pts) >= N:
                break
            x = (j + 0.5) * dx + (i % 2) * dx / 2
            y = (i + 0.5) * dy
            if 0 < x < 1 and 0 < y < 1:
                pts.append((x, y, r))
    if len(pts) < N:
        # Fill remaining with grid
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                x, y = (i + 0.5) / side, (j + 0.5) / side
                if not any(np.allclose([x, y], [p[0], p[1]], atol=0.05) for p in pts):
                    pts.append((x, y, r))
    return np.array(pts[:N])

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    r = 0.05
    pts = []
    for _ in range(N * 10):
        if len(pts) >= N:
            break
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        if all(np.sqrt((x - p[0])**2 + (y - p[1])**2) > 2*r for p in pts):
            pts.append((x, y, r))
    # Fill any remaining
    while len(pts) < N:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        pts.append((x, y, r))
    return np.array(pts[:N])

def _center_ring_init():
    """One large center circle with rings around it."""
    pts = []
    # Center circle
    r_center = 0.12
    pts.append((0.5, 0.5, r_center))
    # Ring 1: 6 circles
    r1 = 0.07
    for k in range(6):
        angle = k * np.pi / 3
        x = 0.5 + (r_center + r1) * np.cos(angle)
        y = 0.5 + (r_center + r1) * np.sin(angle)
        if 0 < x < 1 and 0 < y < 1:
            pts.append((x, y, r1))
    # Ring 2: 12 circles
    r2 = 0.05
    for k in range(12):
        angle = k * np.pi / 6
        x = 0.5 + (r_center + 2*r1 + r2) * np.cos(angle)
        y = 0.5 + (r_center + 2*r1 + r2) * np.sin(angle)
        if 0 < x < 1 and 0 < y < 1:
            pts.append((x, y, r2))
    # Fill remaining
    while len(pts) < N:
        angle = np.random.random() * 2 * np.pi
        rad = 0.4
        x = 0.5 + rad * np.cos(angle)
        y = 0.5 + rad * np.sin(angle)
        if 0 < x < 1 and 0 < y < 1:
            pts.append((x, y, 0.03))
    return np.array(pts[:N])

def _optimize(init_pts, time_limit=30):
    """Run L-BFGS-B optimization from given initial points."""
    start_time = time.time()
    x = init_pts[:, 0].copy()
    y = init_pts[:, 1].copy()
    r = init_pts[:, 2].copy()
    
    best_score = _score(x, y, r)
    best_params = np.concatenate([x, y, r])
    
    params = np.concatenate([x, y, r])
    
    # L-BFGS-B optimization
    try:
        res = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N + [(0, 1)] * N + [(0.001, 0.5)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() - start_time < time_limit:
            x2, y2, r2 = res.x[:N], res.x[N:2*N], res.x[2*N:]
            s2 = _score(x2, y2, r2)
            if s2 > best_score:
                best_score = s2
                best_params = res.x
    except Exception:
        pass
    
    if time.time() - start_time >= time_limit:
        return best_params, best_score
    
    # Nelder-Mead for further refinement
    try:
        params2 = best_params.copy()
        res2 = minimize(
            _objective,
            params2,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        x3, y3, r3 = res2.x[:N], res2.x[N:2*N], res2.x[2*N:]
        s3 = _score(x3, y3, r3)
        if s3 > best_score:
            best_score = s3
            best_params = res2.x
    except Exception:
        pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try multiple initializations
    inits = [
        _grid_init(),
        _hex_init(),
        _center_ring_init(),
    ]
    for seed in [42, 123, 456]:
        inits.append(_random_init(seed))
    
    best_params = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            params, score = _optimize(init, time_limit=20)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to grid
        pts = _grid_init()
        return np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2]])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:]
    
    # Final validation and clamping
    x = _clamp(x, 0, 1)
    y = _clamp(y, 0, 1)
    r = _clamp(r, 0.001, 0.5)
    
    # Ensure circles are inside unit square
    for i in range(N):
        r[i] = min(r[i], x[i], y[i], 1 - x[i], 1 - y[i])
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
