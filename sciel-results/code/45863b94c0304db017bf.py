"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square."""
    circles = circles.copy()
    for i in range(N):
        r = max(0, circles[i, 2])
        if r < 0.5:
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        else:
            circles[i, 0] = 0.5
            circles[i, 1] = 0.5
        circles[i, 2] = r
    return circles


def _objective_penalty(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Vectorized objective with soft penalty for constraint violations.
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    # Pairwise overlaps
    n = len(r)
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(max(dx * dx + dy * dy, 1e-12))
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps) ** 2
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _get_contact_edges(circles):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None


def _initial_hex():
    """Hexagonal packing initial config."""
    positions = []
    # Try a few hexagonal arrangements
    # 5 columns, 5 rows = 25, plus 1
    cols = 5
    rows = 5
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x <= 1 and y <= 1:
                positions.append([x, y, r])
    # Add one more in the center gap
    positions.append([0.5, 0.5, r])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty and contact graph."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 1: Low penalty, focus on layout, use full pairwise
    penalty = 5.0
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3, None),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty, use contact graph
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        circles = params.reshape(N, 3)
        edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, edges),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4, edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty, refine with updated contact graph
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 500.0
        circles = params.reshape(N, 3)
        edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, edges),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-5, edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Final validation and repair
    circles = best_params.reshape(N, 3)
    circles = _fix_boundaries(circles)
    if not _validate(circles):
        # Try to repair by reducing radii slightly
        for i in range(N):
            circles[i, 2] *= 0.95
        circles = _fix_boundaries(circles)

    return circles, np.sum(circles[:, 2])


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses multi-start optimization with adaptive penalty and Delaunay contact graph.
    """
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin

    # Try multiple initial configurations
    initial_configs = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(42),
        _initial_corner_weighted(123),
    ]

    best_circles = None
    best_score = 0.0

    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break

        circles, score = _optimize_single(config, deadline)
        if score > best_score:
            best_score = score
            best_circles = circles

        # If we have time, try a few more random starts
        if i < 2 and time.time() < deadline - 10:
            for seed in [7, 17, 23]:
                if time.time() > deadline - 5:
                    break
                config2 = _initial_corner_weighted(seed)
                circles2, score2 = _optimize_single(config2, deadline)
                if score2 > best_score:
                    best_score = score2
                    best_circles = circles2

    if best_circles is None:
        # Fallback
        best_circles = _initial_grid()

    # Final validation
    if not _validate(best_circles):
        for i in range(N):
            best_circles[i, 2] *= 0.9
        best_circles = _fix_boundaries(best_circles)

    return best_circles

# EVOLVE-BLOCK-END
