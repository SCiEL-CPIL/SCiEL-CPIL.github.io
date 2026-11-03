"""Circle packing in unit square: center-only optimization with geometry-derived radii.

Structural change: instead of optimizing 78 variables (x,y,r per circle), we
optimize only the 52 center coordinates. Radii are derived from the geometry:
each circle's radius is the minimum distance to the boundary and to half the
distance to every other circle center. This eliminates the radius optimization
variables and their associated non-differentiability issues.

The objective maximizes sum of derived radii, which naturally encourages
dense, space-filling configurations where every circle is "squeezed" by its
neighbors and/or the boundary.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # Leave 10s safety margin
START = time.time()


def _elapsed():
    return time.time() - START


def _derive_radii(x, y):
    """
    Derive the maximum valid radius for each circle given center positions.
    r[i] = min(distance to boundary, min over j!=i of dist(i,j)/2)
    """
    # Distance to each boundary
    d_left = x
    d_right = 1.0 - x
    d_bottom = y
    d_top = 1.0 - y
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Distance to other circles (half the pairwise distance)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    # For each circle, the constraint from neighbor j is dist(i,j)/2
    # So r[i] <= dist(i,j)/2 for all j
    r_from_neighbors = np.min(dist, axis=1) / 2.0
    
    # Final radius is the minimum of boundary distance and neighbor constraint
    r = np.minimum(d_bound, r_from_neighbors)
    r = np.maximum(r, 1e-8)
    
    return r


def _sum_radii(params):
    """
    Objective to minimize: -sum of derived radii.
    params: 52-dim vector [x0, y0, x1, y1, ..., x25, y25]
    """
    x = params[0::2]
    y = params[1::2]
    
    # Clip to valid ranges (slightly inside to avoid boundary issues)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    
    r = _derive_radii(x, y)
    return -np.sum(r)


def _valid(x, y, r, eps=1e-6):
    """Check if configuration is valid."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _grid_init_5x5_plus1(rng):
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_4x7(rng):
    cols, rows = 4, 7
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_6x5(rng):
    cols, rows = 6, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    return x, y


def _hex_init(rng):
    """Hexagonal lattice initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.19
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.15 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.15 + row * spacing * 0.866
        if x[i] > 0.9:
            x[i] = 0.9
        if y[i] > 0.9:
            y[i] = 0.9
    return x, y


def _optimize_centers(x, y, method='L-BFGS-B'):
    """Optimize center positions to maximize sum of derived radii."""
    params = np.zeros(2 * N)
    params[0::2] = x
    params[1::2] = y
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    if method == 'L-BFGS-B':
        try:
            res = minimize(_sum_radii, params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 5000})
        except Exception:
            return x, y
    elif method == 'SLSQP':
        try:
            res = minimize(_sum_radii, params, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10})
        except Exception:
            return x, y
    else:
        return x, y
    
    v = res.x
    xx = np.clip(v[0::2], 0.0, 1.0)
    yy = np.clip(v[1::2], 0.0, 1.0)
    
    return xx, yy


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x7,
        _grid_init_6x5,
        _random_init,
        _hex_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        x, y = init_func(rng)
        
        for restart in range(5):
            if _elapsed() > DEADLINE - 10.0:
                break
            
            # Add small jitter
            x0 = np.clip(x + rng.normal(0, 0.01, N), 0.01, 0.99)
            y0 = np.clip(y + rng.normal(0, 0.01, N), 0.01, 0.99)
            
            # First pass: L-BFGS-B
            xx, yy = _optimize_centers(x0, y0, method='L-BFGS-B')
            rr = _derive_radii(xx, yy)
            
            # Second pass: SLSQP refinement
            if _elapsed() < DEADLINE - 15.0:
                xx2, yy2 = _optimize_centers(xx, yy, method='SLSQP')
                rr2 = _derive_radii(xx2, yy2)
                
                if _valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])
            
            # Check first pass result
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
