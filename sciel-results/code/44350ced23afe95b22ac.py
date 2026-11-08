"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Structural change: vectorized soft-plus penalty with L-BFGS-B optimization.
Uses smooth log-sum-exp surrogates to avoid non-differentiable kinks, and
fully vectorized pairwise distance computation for fast gradient evaluation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 115.0  # Safety margin
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
        return False
    return True

def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf

def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])

def _softplus(v, T):
    """Smooth max(0, v) via log-sum-exp."""
    a = v / T
    # Numerically stable softplus
    return T * np.log1p(np.exp(np.clip(a, -50, 50)))

def _penalty_objective(params, T, weight):
    """Vectorized soft-plus penalty objective. Returns (value, gradient)."""
    c = params.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    # Wall violations
    w1 = r - x        # left wall
    w2 = x + r - 1    # right wall
    w3 = r - y        # bottom wall
    w4 = y + r - 1    # top wall
    
    # Pairwise overlaps (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    s = r[:, None] + r[None, :]
    overlap = s - d  # positive means overlap
    
    # Only compute softplus for pairs that might be close (optimization)
    # For N=26, full 26x26 is fine
    pen_wall = _softplus(w1, T) + _softplus(w2, T) + _softplus(w3, T) + _softplus(w4, T)
    pen_overlap = _softplus(overlap, T)
    # Use upper triangle to avoid double counting
    tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pen_overlap = np.sum(pen_overlap[tri_mask])
    
    pen_wall = np.sum(pen_wall)
    total_pen = pen_wall + pen_overlap
    obj = -np.sum(r) + weight * total_pen
    
    # Gradient
    grad = np.zeros_like(params)
    # Wall gradients: d/dx of softplus(r-x) = -sigma(r-x)/T * T = -sigma(r-x)
    # sigma(a) = 1/(1+exp(-a))
    def _sigma(a):
        return 1.0 / (1.0 + np.exp(-np.clip(a, -50, 50)))
    
    s1 = _sigma((r - x) / T)
    s2 = _sigma((x + r - 1) / T)
    s3 = _sigma((r - y) / T)
    s4 = _sigma((y + r - 1) / T)
    
    grad[:, 0] += -1.0 + weight * (-s1 + s2)
    grad[:, 1] += -1.0 + weight * (-s3 + s4)
    grad[:, 2] += -1.0 + weight * (s1 + s2 + s3 + s4)
    
    # Pairwise overlap gradients
    # d/dx_i of softplus(s_ij - d_ij) where s_ij = r_i + r_j, d_ij = ||p_i - p_j||
    # = sigma((s_ij - d_ij)/T) * (r_i + r_j - d_ij derivative part)
    # d/dx_i d_ij = -(x_i - x_j)/d_ij
    # d/dr_i d_ij contribution: +1 (since s_ij increases)
    
    sigma_ov = _sigma(overlap / T)
    
    # For each pair (i,j), i<j:
    # d obj / dx_i += weight * sigma_ov[i,j] * (-(x_i - x_j)/d_ij)  [from -d_ij term]
    # Actually: d/dx_i [softplus(s-d)] = sigma((s-d)/T) * d/dx_i(s-d) = sigma * (0 - d/dx_i d)
    # d/dx_i d = (x_i - x_j)/d, so d/dx_i (s-d) = -(x_i - x_j)/d
    # d/dr_i d = 1 (s increases by 1)
    
    # Vectorized: for each i, sum over j != i
    # ux[i,j] = (x_i - x_j)/d[i,j]
    ux = dx / d
    uy = dy / d
    
    # grad_x_i = weight * sum_j sigma_ov[i,j] * (-ux[i,j]) for j != i
    # But we only want upper triangle counted once... actually for gradient we need all j
    # Let's just use full matrix and subtract diagonal
    gx = -np.sum(sigma_ov * ux, axis=1)
    gy = -np.sum(sigma_ov * uy, axis=1)
    gr = np.sum(sigma_ov, axis=1)
    
    # Subtract diagonal (sigma_ov[i,i] * ux[i,i] = sigma * 0 = 0, so fine)
    # But gr includes sigma_ov[i,i] which corresponds to d=1.0, overlap[i,i] = 2*r_i - 1
    # We should zero out diagonal
    np.fill_diagonal(sigma_ov, 0.0)
    gx = -np.sum(sigma_ov * ux, axis=1)
    gy = -np.sum(sigma_ov * uy, axis=1)
    gr = np.sum(sigma_ov, axis=1)
    
    grad[:, 0] += weight * gx
    grad[:, 1] += weight * gy
    grad[:, 2] += weight * gr
    
    return obj, grad

def _optimize_stage(c0, T, weight, max_iter=200):
    """Run one L-BFGS-B stage with given temperature and weight."""
    x0 = c0.flatten()
    
    def _f(p):
        val, _ = _penalty_objective(p, T, weight)
        return val
    
    def _g(p):
        _, g = _penalty_objective(p, T, weight)
        return g
    
    try:
        result = minimize(
            _f, x0, jac=_g,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        c_new = result.x.reshape(N, 3)
        c_new = _clamp(c_new)
        return c_new
    except Exception:
        return c0

def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _seed_hex():
    """Hexagonal packing initialization."""
    positions = []
    r = 0.18
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(5):
        for col in range(5):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y, r))
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r_small = RNG.uniform(0.05, 0.1)
        positions.append((x, y, r_small))
    return np.array(positions[:N])

def _seed_random():
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.03, 0.1, N)
    return np.column_stack([x, y, r])

def _seed_corner_heavy():
    """Initialize with larger circles in corners, smaller in center."""
    positions = []
    radii = []
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.13)
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        positions.append((cx, cy))
        radii.append(0.1)
    positions.append((0.5, 0.5))
    radii.append(0.12)
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.04, 0.08)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])

def _seed_symmetric():
    """Symmetric 5x5 grid with center emphasis."""
    positions = []
    # 5x5 grid = 25 circles, add 1 more
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            # Larger in corners, smaller in center
            dist_center = np.hypot(x - 0.5, y - 0.5)
            r = 0.09 + 0.04 * dist_center
            positions.append((x, y, r))
    # 26th circle: small in a gap
    positions.append((0.5, 0.5, 0.05))
    return np.array(positions[:N])

def construct():
    t0 = time.time()
    
    best_c = None
    best_val = -np.inf
    
    # Phase 1: Multi-start with diverse initializations using L-BFGS-B
    initializers = [_seed_grid, _seed_hex, _seed_corner_heavy, _seed_random, _seed_symmetric]
    
    for init_func in initializers:
        if time.time() > DEADLINE - 20.0:
            break
        
        c0 = init_func()
        c0 = _clamp(c0)
        
        # Annealing schedule: start with high T (loose), decrease T (tight)
        # Increase weight to enforce constraints more strongly
        stages = [
            (0.05, 50.0, 150),
            (0.02, 100.0, 200),
            (0.008, 200.0, 200),
            (0.003, 500.0, 300),
            (0.001, 1000.0, 300),
        ]
        
        c_current = c0.copy()
        for T, weight, max_iter in stages:
            if time.time() > DEADLINE - 8.0:
                break
            c_current = _optimize_stage(c_current, T, weight, max_iter)
        
        # Evaluate
        val = _score(c_current)
        if val > best_val:
            best_val = val
            best_c = c_current.copy()
    
    # Phase 2: Additional random starts if time permits
    n_extra = 8
    for i in range(n_extra):
        if time.time() > DEADLINE - 15.0:
            break
        
        c0 = _seed_random()
        c0 = _clamp(c0)
        
        stages = [
            (0.04, 50.0, 100),
            (0.015, 150.0, 150),
            (0.005, 300.0, 200),
            (0.002, 800.0, 250),
        ]
        
        c_current = c0.copy()
        for T, weight, max_iter in stages:
            if time.time() > DEADLINE - 8.0:
                break
            c_current = _optimize_stage(c_current, T, weight, max_iter)
        
        val = _score(c_current)
        if val > best_val:
            best_val = val
            best_c = c_current.copy()
    
    # Phase 3: Final high-precision polish on best result
    if best_c is not None and time.time() < DEADLINE - 10.0:
        c_polished = best_c.copy()
        polish_stages = [
            (0.001, 1000.0, 300),
            (0.0004, 2000.0, 400),
            (0.0001, 5000.0, 500),
        ]
        for T, weight, max_iter in polish_stages:
            if time.time() > DEADLINE - 5.0:
                break
            c_polished = _optimize_stage(c_polished, T, weight, max_iter)
        
        val = _score(c_polished)
        if val > best_val:
            best_val = val
            best_c = c_polished.copy()
    
    # Final safety: ensure valid output
    if best_c is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])
    
    return best_c
