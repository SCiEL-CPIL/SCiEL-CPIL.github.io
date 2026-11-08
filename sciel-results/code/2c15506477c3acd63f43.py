"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 145.0  # safety margin vs 150s kill


def _validate(arr):
    if arr is None or arr.shape[0] != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0) or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False
    # bounding rectangle
    xmin, xmax = np.min(x - r), np.max(x + r)
    ymin, ymax = np.min(y - r), np.max(y + r)
    W = xmax - xmin
    H = ymax - ymin
    if 2.0 * (W + H) > 4.0 + 1e-5:
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            rs = r[i] + r[j]
            if d2 < rs * rs - 1e-7:
                return False
    return True


def _compute_radii(xs, ys, W, H):
    """Compute the maximum radius for each circle given positions,
    constrained by boundaries and other circles.
    For each circle i, r_i = min over all constraints of the limiting distance."""
    n = len(xs)
    radii = np.full(n, np.inf)
    # boundary constraints
    radii = np.minimum(radii, xs)          # left
    radii = np.minimum(radii, W - xs)      # right
    radii = np.minimum(radii, ys)          # bottom
    radii = np.minimum(radii, H - ys)      # top
    # circle-circle constraints: r_i + r_j <= dist_ij
    # This is coupled, but we can iterate or use a simpler bound.
    # For now, use pairwise: r_i <= (dist_ij - r_j) but that's circular.
    # Better: for each pair, the constraint is r_i + r_j <= d_ij.
    # We can solve this as: for each i, r_i <= min_j (d_ij - r_j), but r_j unknown.
    # Simple approach: iterate a few times.
    for _ in range(20):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                # r_i + r_j <= d  =>  r_i <= d - r_j
                limit = d - radii[j]
                if limit < radii[i]:
                    radii[i] = limit
    radii = np.maximum(radii, 0.0)
    return radii


def _sum_radii(xs, ys, W, H):
    radii = _compute_radii(xs, ys, W, H)
    return np.sum(radii)


def _objective_fixed_box(z, W, H):
    """z = [x1, y1, x2, y2, ..., xn, yn]. Returns -sum of radii."""
    n = N
    xs = z[:n]
    ys = z[n:2*n]
    # Keep positions within [0, W] x [0, H]
    xs = np.clip(xs, 0, W)
    ys = np.clip(ys, 0, H)
    s = _sum_radii(xs, ys, W, H)
    return -s


def _grid_initialization(cols, rows, W, H, n, jitter=0.0, seed=0):
    """Place n circles in a cols x rows grid within W x H."""
    rng = np.random.default_rng(seed)
    xs = np.zeros(n)
    ys = np.zeros(n)
    cw = W / cols
    ch = H / rows
    for k in range(n):
        ci = k % cols
        ri = k // cols
        xs[k] = (ci + 0.5) * cw
        ys[k] = (ri + 0.5) * ch
    if jitter > 0:
        xs += rng.normal(0, jitter, n)
        ys += rng.normal(0, jitter, n)
        xs = np.clip(xs, 0.01 * W, 0.99 * W)
        ys = np.clip(ys, 0.01 * H, 0.99 * H)
    return xs, ys


def _random_initialization(W, H, n, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1 * W, 0.9 * W, n)
    ys = rng.uniform(0.1 * H, 0.9 * H, n)
    return xs, ys


def _optimize_for_aspect(aspect, init_type='grid', cols=7, rows=3, seed=0, max_iter=5000):
    """Optimize for a given aspect ratio W/H with W+H=2."""
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    
    if init_type == 'grid':
        xs, ys = _grid_initialization(cols, rows, W, H, N, jitter=0.02 * min(W, H), seed=seed)
    else:
        xs, ys = _random_initialization(W, H, N, seed=seed)
    
    z0 = np.concatenate([xs, ys])
    
    # Bounds for positions
    bounds = [(0, W)] * N + [(0, H)] * N
    
    try:
        res = minimize(
            _objective_fixed_box,
            z0,
            args=(W, H),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        z_opt = res.x
        xs_opt = z_opt[:N]
        ys_opt = z_opt[N:2*N]
        xs_opt = np.clip(xs_opt, 0, W)
        ys_opt = np.clip(ys_opt, 0, H)
    except Exception:
        return None, 0.0
    
    radii = _compute_radii(xs_opt, ys_opt, W, H)
    total = np.sum(radii)
    
    # Build result array
    arr = np.column_stack([xs_opt, ys_opt, radii])
    
    # Validate
    if not _validate(arr):
        # Try to fix by reducing radii slightly
        for factor in [0.99, 0.98, 0.97, 0.95, 0.9, 0.8]:
            arr[:, 2] = radii * factor
            if _validate(arr):
                total = np.sum(arr[:, 2])
                break
        else:
            return None, 0.0
    
    return arr, total


def construct():
    best_arr = None
    best_val = -1.0
    
    # Candidate configurations: (aspect, init_type, cols, rows)
    configs = [
        (7.0/3.0, 'grid', 7, 3),
        (1.5, 'grid', 7, 3),
        (1.2, 'grid', 7, 3),
        (1.5, 'grid', 6, 4),
        (1.0, 'grid', 5, 5),
        (2/3, 'grid', 4, 6),
        (3/7, 'grid', 3, 7),
        (7.0/3.0, 'random', 0, 0),
        (1.5, 'random', 0, 0),
        (1.2, 'random', 0, 0),
        (1.0, 'random', 0, 0),
        (1.3, 'grid', 7, 3),
        (1.4, 'grid', 7, 3),
        (1.6, 'grid', 7, 3),
        (1.1, 'grid', 7, 3),
    ]
    
    for idx, (aspect, init_type, cols, rows) in enumerate(configs):
        if time.time() > _DEADLINE - 10.0:
            break
        arr, val = _optimize_for_aspect(aspect, init_type, cols, rows, seed=idx, max_iter=3000)
        if arr is not None and val > best_val:
            best_val = val
            best_arr = arr
    
    # Fallback: simple grid
    if best_arr is None or best_val <= 0:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
        best_val = np.sum(best_arr[:, 2])
    
    return best_arr
