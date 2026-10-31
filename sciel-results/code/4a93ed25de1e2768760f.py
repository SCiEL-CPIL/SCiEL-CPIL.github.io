"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural change: Multi-start L-BFGS-B with a penalty formulation that
simultaneously optimizes radii and positions, plus strict validity repair.
This overcomes the local minima of the max-radius-per-circle formulation by
allowing circles to "push" against each other through gradient-based search.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute pairwise index pairs (i < j)
_PAIRS = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
_PI = _PAIRS[:, 0]
_PJ = _PAIRS[:, 1]
N_PAIRS = len(_PAIRS)


def _penalty_objective(params, w, h, lam):
    """
    params: [r_1..r_N, x_1..x_N, y_1..y_N]
    Objective: -sum(r) + lam * (sum of constraint violations)
    Constraints:
      1. r_i >= 0
      2. x_i in [r_i, w - r_i]
      3. y_i in [r_i, h - r_i]
      4. dist(i,j) >= r_i + r_j
    """
    r = params[:N]
    x = params[N:2*N]
    y = params[2*N:3*N]

    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)

    # Penalty for negative radii
    neg_r = np.maximum(0.0, -r)
    obj += lam * np.sum(neg_r ** 2)

    # Penalty for boundary violations
    # x_i < r_i or x_i > w - r_i
    viol_x_low = np.maximum(0.0, r - x)
    viol_x_high = np.maximum(0.0, x - (w - r))
    viol_y_low = np.maximum(0.0, r - y)
    viol_y_high = np.maximum(0.0, y - (h - r))
    obj += lam * (np.sum(viol_x_low**2) + np.sum(viol_x_high**2) +
                  np.sum(viol_y_low**2) + np.sum(viol_y_high**2))

    # Penalty for overlap
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[_PI] + r[_PJ]
    viol_overlap = np.maximum(0.0, min_dist - dist)
    obj += lam * np.sum(viol_overlap ** 2)

    return obj


def _penalty_grad(params, w, h, lam):
    """Gradient of penalty objective."""
    r = params[:N]
    x = params[N:2*N]
    y = params[2*N:3*N]

    grad = np.zeros(3 * N)

    # d(-sum(r))/dr = -1
    grad[:N] = -1.0

    # Gradient of neg_r penalty: 2*lam*neg_r*(-1) for neg_r>0
    neg_r = np.maximum(0.0, -r)
    grad[:N] += lam * (-2.0 * neg_r)

    # Boundary penalties
    viol_x_low = np.maximum(0.0, r - x)
    viol_x_high = np.maximum(0.0, x - (w - r))
    viol_y_low = np.maximum(0.0, r - y)
    viol_y_high = np.maximum(0.0, y - (h - r))

    # d/dx: -2*lam*viol_x_low + 2*lam*viol_x_high
    grad[N:2*N] += lam * (-2.0 * viol_x_low + 2.0 * viol_x_high)
    # d/dy
    grad[2*N:3*N] += lam * (-2.0 * viol_y_low + 2.0 * viol_y_high)

    # d/dr from boundary: 2*lam*viol_x_low + 2*lam*viol_x_high + 2*lam*viol_y_low + 2*lam*viol_y_high
    grad[:N] += lam * (2.0 * viol_x_low + 2.0 * viol_x_high +
                       2.0 * viol_y_low + 2.0 * viol_y_high)

    # Overlap penalty gradient
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    dist = np.sqrt(dx**2 + dy**2)
    dist_safe = np.maximum(dist, 1e-12)
    min_dist = r[_PI] + r[_PJ]
    viol = np.maximum(0.0, min_dist - dist)

    # d/dx_i for pair (i,j): 2*lam*viol * (x_i - x_j)/dist
    # Use accumulation
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)

    # For each pair, contribution to i and j
    factor = 2.0 * lam * viol / dist_safe
    gx[_PI] += factor * dx
    gx[_PJ] -= factor * dx
    gy[_PI] += factor * dy
    gy[_PJ] -= factor * dy
    gr[_PI] += 2.0 * lam * viol
    gr[_PJ] += 2.0 * lam * viol

    grad[N:2*N] += gx
    grad[2*N:3*N] += gy
    grad[:N] += gr

    return grad


def _repair_and_compute(params, w, h):
    """
    Given optimized params, repair to valid configuration and compute sum of radii.
    Returns (valid_params, sum_r) or (None, 0) if invalid.
    """
    r = params[:N].copy()
    x = params[N:2*N].copy()
    y = params[2*N:3*N].copy()

    # Clip radii to non-negative
    r = np.maximum(r, 0.0)

    # Iterative repair: shrink radii to resolve overlaps
    for _ in range(50):
        # Check and fix boundary violations
        for i in range(N):
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            if r[i] < 0:
                r[i] = 0.0

        # Check overlaps
        max_shrink = 0.0
        for k in range(N_PAIRS):
            i, j = _PI[k], _PJ[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                # Shrink both equally
                shrink = (min_dist - dist) / 2.0 + 1e-9
                max_shrink = max(max_shrink, shrink)

        if max_shrink < 1e-12:
            break

        # Apply shrink: reduce radii proportionally
        # Simple approach: scale all radii by (1 - max_shrink / max_r)
        max_r = np.max(r)
        if max_r > 0:
            scale = max(0.0, 1.0 - max_shrink / (max_r + 1e-12))
            r *= scale
        else:
            break

    # Final validation
    r = np.maximum(r, 0.0)
    for i in range(N):
        r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
        r[i] = max(r[i], 0.0)

    # Check all constraints
    for i in range(N):
        if r[i] < 0 or x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9 or \
           y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
            # Try to fix
            r[i] = min(r[i], max(0.0, x[i] - 1e-9), max(0.0, w - x[i] - 1e-9),
                       max(0.0, y[i] - 1e-9), max(0.0, h - y[i] - 1e-9))

    for k in range(N_PAIRS):
        i, j = _PI[k], _PJ[k]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[j]
        if dist < min_dist - 1e-9:
            # Still overlapping; shrink more aggressively
            shrink = (min_dist - dist) / 2.0 + 1e-8
            r[i] = max(0.0, r[i] - shrink)
            r[j] = max(0.0, r[j] - shrink)

    # Final check
    for k in range(N_PAIRS):
        i, j = _PI[k], _PJ[k]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[j]
        if dist < min_dist - 1e-7:
            return None, 0.0

    for i in range(N):
        if r[i] < -1e-9 or x[i] < r[i] - 1e-7 or x[i] > w - r[i] + 1e-7 or \
           y[i] < r[i] - 1e-7 or y[i] > h - r[i] + 1e-7:
            return None, 0.0

    sum_r = float(np.sum(r))
    if sum_r <= 0:
        return None, 0.0

    return (r, x, y), sum_r


def _grid_init(w, h, n_cols=None):
    """Generate a grid-based initial configuration."""
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0 else 5
    n_rows = int(np.ceil(N / n_cols))

    # Adjust to fit in rectangle
    cell_w = w / n_cols
    cell_h = h / n_rows

    params = np.zeros(3 * N)
    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if count >= N:
                break
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            r = min(cell_w, cell_h) / 2.0 * 0.95
            params[count] = r
            params[N + count] = cx
            params[2*N + count] = cy
            count += 1

    return params


def _random_init(w, h):
    """Random initial configuration with small radii."""
    params = np.zeros(3 * N)
    for i in range(N):
        r = random.uniform(0.02, min(w, h) * 0.15)
        x = random.uniform(r, w - r)
        y = random.uniform(r, h - r)
        params[i] = r
        params[N + i] = x
        params[2*N + i] = y
    return params


def _optimize_one(w, h, init_params, time_budget=10.0):
    """Run L-BFGS-B on one starting point."""
    start = time.time()

    # Start with moderate penalty, increase if needed
    lam = 10.0
    best_params = init_params.copy()
    best_sum_r = 0.0

    for attempt in range(5):
        if time.time() - start > time_budget:
            break

        try:
            res = minimize(
                _penalty_objective,
                init_params,
                args=(w, h, lam),
                jac=_penalty_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                         'maxls': 50}
            )
            repaired, sum_r = _repair_and_compute(res.x, w, h)
            if repaired is not None and sum_r > best_sum_r:
                best_sum_r = sum_r
                best_params = res.x.copy()

            # If still invalid or low quality, increase penalty
            if sum_r < best_sum_r * 0.95:
                lam *= 2.0
                # Perturb initial point
                init_params = best_params + np.random.randn(3*N) * 0.01
        except Exception:
            lam *= 2.0
            init_params = best_params + np.random.randn(3*N) * 0.05

    return best_params, best_sum_r


def _build_result(r, x, y, w, h):
    """Build output array with tiny shrink for strict disjointness."""
    out = np.zeros((N, 3))
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = r * (1.0 - 1e-7)
    return out


def _grid_fallback():
    """Simple valid packing: 5x5 grid in a square, drop 4 corners."""
    w = h = 1.0
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
    r = (0.5 / side) * (1.0 - 1e-4)
    out = np.zeros((N, 3))
    for i, (cx, cy) in enumerate(cells):
        out[i, 0] = cx * w
        out[i, 1] = cy * h
        out[i, 2] = r
    return out


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = -np.inf

    def _consider(r, x, y, w, h):
        nonlocal best_result, best_score
        sum_r = float(np.sum(r))
        if sum_r > best_score:
            best_score = sum_r
            best_result = _build_result(r, x, y, w, h)

    # Try multiple aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]

    # Sort by time budget per aspect
    n_aspects = len(aspects)
    time_per_aspect = (DEADLINE - time.time() - 10.0) / n_aspects
    time_per_aspect = max(time_per_aspect, 5.0)

    for aspect in aspects:
        if time.time() > DEADLINE - 10.0:
            break

        w = aspect
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            continue

        # Multiple starting points per aspect
        n_starts = 3
        time_per_start = time_per_aspect / n_starts
        time_per_start = max(time_per_start, 2.0)

        for s in range(n_starts):
            if time.time() > DEADLINE - 10.0:
                break

            if s == 0:
                init = _grid_init(w, h)
            elif s == 1:
                init = _random_init(w, h)
            else:
                # Perturbed grid
                init = _grid_init(w, h)
                init += np.random.randn(3*N) * 0.05
                # Ensure radii non-negative
                init[:N] = np.maximum(init[:N], 0.01)

            try:
                params, sum_r = _optimize_one(w, h, init, time_budget=time_per_start)
                if sum_r > 0:
                    repaired, final_sum = _repair_and_compute(params, w, h)
                    if repaired is not None:
                        r, x, y = repaired
                        _consider(r, x, y, w, h)
            except Exception:
                pass

    # If no valid result, use fallback
    if best_result is None:
        best_result = _grid_fallback()

    return best_result
