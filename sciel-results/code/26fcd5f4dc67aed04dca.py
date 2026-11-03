import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _elapsed(start):
    return time.time() - start


def _penalized(params, w_overlap=200.0, w_bound=200.0):
    """
    Vectorized penalty-based objective.
    Minimize: -sum(r) + w_bound * boundary_penalty + w_overlap * overlap_penalty
    params: shape (N, 3) = [x, y, r]
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]

    obj = -np.sum(r)

    # Boundary penalty: circles must stay inside [0,1]^2
    # Violation = max(0, r - min(x, 1-x, y, 1-y))
    min_dist_to_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    boundary_violation = np.maximum(0, r - min_dist_to_boundary)
    penalty_boundary = np.sum(boundary_violation)

    # Overlap penalty: d_ij >= r_i + r_j
    # Compute all pairwise distances
    penalty_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if d < min_dist:
                penalty_overlap += (min_dist - d) ** 2

    return obj + w_bound * penalty_boundary + w_overlap * penalty_overlap


def _valid(params, tol=1e-5):
    """Check if configuration is valid (no overlaps, inside square)."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]

    # Check boundaries
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False

    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False

    return True


def _score(params):
    """Return sum of radii if valid, else -inf."""
    if _valid(params):
        return np.sum(params[:, 2])
    return -np.inf


def _fix_to_boundary(params):
    """Shrink circles that violate boundary constraints."""
    params = params.copy()
    for i in range(N):
        min_dist = min(params[i, 0], 1 - params[i, 0], params[i, 1], 1 - params[i, 1])
        if params[i, 2] > min_dist:
            params[i, 2] = max(min_dist, 0.0)
    return params


def _fix_overlaps(params):
    """Iteratively shrink overlapping circles to resolve violations."""
    params = params.copy()
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = params[i, 0] - params[j, 0]
                dy = params[i, 1] - params[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_dist = params[i, 2] + params[j, 2]
                if d < min_dist - 1e-7:
                    # Shrink the smaller circle
                    if params[i, 2] <= params[j, 2]:
                        new_r = max(d - params[j, 2], 0.0)
                        if params[i, 2] > new_r:
                            params[i, 2] = new_r
                            changed = True
                    else:
                        new_r = max(d - params[i, 2], 0.0)
                        if params[j, 2] > new_r:
                            params[j, 2] = new_r
                            changed = True
        if not changed:
            break
    return params


def _grid_init_5x5_plus1():
    """5x5 grid (25 circles) plus one extra circle in a gap."""
    cells = []
    side = 5
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    # Add 26th circle at center, smaller radius
    cells.append([0.5, 0.5, 0.05])
    return np.array(cells[:N])


def _grid_init_4x7():
    """4x7 grid (28 cells, use first 26)."""
    cells = []
    nx, ny = 4, 7
    for i in range(nx):
        for j in range(ny):
            if len(cells) < N:
                x = (i + 0.5) / nx
                y = (j + 0.5) / ny
                r = 0.5 / max(nx, ny) - 1e-4
                cells.append([x, y, r])
    return np.array(cells[:N])


def _grid_init_7x4():
    """7x4 grid (28 cells, use first 26)."""
    cells = []
    nx, ny = 7, 4
    for i in range(nx):
        for j in range(ny):
            if len(cells) < N:
                x = (i + 0.5) / nx
                y = (j + 0.5) / ny
                r = 0.5 / max(nx, ny) - 1e-4
                cells.append([x, y, r])
    return np.array(cells[:N])


def _random_init(rng):
    """Random initial configuration with small circles."""
    params = np.zeros((N, 3))
    params[:, 0] = rng.uniform(0.1, 0.9, N)
    params[:, 1] = rng.uniform(0.1, 0.9, N)
    params[:, 2] = rng.uniform(0.02, 0.15, N)
    return params


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin under 150s kill

    best_params = None
    best_score = -np.inf

    rng = np.random.default_rng(42)

    # List of initializations to try
    inits = [
        _grid_init_5x5_plus1(),
        _grid_init_4x7(),
        _grid_init_7x4(),
    ]

    # Add a few random inits
    for _ in range(5):
        inits.append(_random_init(rng))

    for init in inits:
        if _elapsed(start_time) > deadline:
            break

        # Try multiple restarts from each init
        for restart in range(3):
            if _elapsed(start_time) > deadline:
                break

            params = init.copy()
            if restart > 0:
                noise = 0.03
                params[:, 0] += rng.uniform(-noise, noise, N)
                params[:, 1] += rng.uniform(-noise, noise, N)
                params[:, 2] += rng.uniform(-noise, noise, N)
                params[:, 0] = np.clip(params[:, 0], 0.02, 0.98)
                params[:, 1] = np.clip(params[:, 1], 0.02, 0.98)
                params[:, 2] = np.clip(params[:, 2], 0.01, 0.4)

            # Optimize with L-BFGS-B
            try:
                res = minimize(
                    _penalized,
                    params,
                    method='L-BFGS-B',
                    bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.45)] * N,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                candidate = res.x.reshape(N, 3)
                candidate[:, 2] = np.maximum(candidate[:, 2], 0.001)
            except Exception:
                continue

            # Fix to boundary and overlaps
            candidate = _fix_to_boundary(candidate)
            candidate = _fix_overlaps(candidate)

            sc = _score(candidate)
            if sc > best_score:
                best_score = sc
                best_params = candidate

    # If we have time, do more random restarts
    while _elapsed(start_time) < deadline:
        params = _random_init(rng)
        try:
            res = minimize(
                _penalized,
                params,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.45)] * N,
                options={'maxiter': 200, 'ftol': 1e-7}
            )
            candidate = res.x.reshape(N, 3)
            candidate[:, 2] = np.maximum(candidate[:, 2], 0.001)
        except Exception:
            continue

        candidate = _fix_to_boundary(candidate)
        candidate = _fix_overlaps(candidate)

        sc = _score(candidate)
        if sc > best_score:
            best_score = sc
            best_params = candidate

    # Final result
    if best_params is None:
        # Fallback: simple grid
        best_params = _grid_init_5x5_plus1()
        best_params = _fix_to_boundary(best_params)
        best_params = _fix_overlaps(best_params)

    return best_params
