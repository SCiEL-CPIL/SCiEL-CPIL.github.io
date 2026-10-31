"""Circle packing in perimeter-4 rectangle: SLSQP refinement of a grid seed."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 150.0
_START = None


def _rect_dims(w):
    return w, 2.0 - w


def _objective(x, w):
    # x layout: [r_0..r_{N-1}, cx_0..cx_{N-1}, cy_0..cy_{N-1}]
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    return -np.sum(r)


def _constraints(w):
    h = 2.0 - w
    cons = []
    # r_i >= eps
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': (lambda i=i: x_i_r(i) - 1e-4)})
    return cons


def _make_constraints(w, h):
    cons = []
    # r_i >= 1e-4
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - 1e-4,
        })
    # cx_i - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[N + i] - x[i],
        })
    # (w - cx_i) - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (w - x[N + i]) - x[i],
        })
    # cy_i - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2 * N + i] - x[i],
        })
    # (h - cy_i) - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (h - x[2 * N + i]) - x[i],
        })
    # pairwise: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })
    return cons


def _grid_seed(w, h):
    # Place N circles in a grid inside rectangle, radius = min gap / 2
    # Choose grid cols x rows with cols*rows >= N
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        # cell sizes
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999  # slight shrink for feasibility
        # build positions
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        # quick feasibility check
        ok = True
        for i in range(N):
            ri = x[i]
            cx = x[N + i]
            cy = x[2 * N + i]
            if cx < ri - 1e-9 or cx > w - ri + 1e-9 or cy < ri - 1e-9 or cy > h - ri + 1e-9:
                ok = False
                break
        if ok:
            for i in range(N):
                for j in range(i + 1, N):
                    d2 = (x[N + i] - x[N + j]) ** 2 + (x[2 * N + i] - x[2 * N + j]) ** 2
                    if d2 < (x[i] + x[j]) ** 2 - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()
    if best_seed is None:
        # fallback: tiny circles
        best_seed = np.zeros(3 * N)
        best_seed[:N] = 1e-4
        best_seed[N:2 * N] = w / 2.0
        best_seed[2 * N:3 * N] = h / 2.0
    return best_seed


def _is_feasible(x, w, tol=1e-6):
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))


def construct():
    global _START
    if _START is None:
        _START = time.time()

    # Try a few aspect ratios
    best_result = None
    best_val = -1e9

    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7]
    # w + h = 2, perimeter 4
    for w in aspect_ratios:
        if time.time() - _START > _DEADLINE - 10:
            break
        if w <= 0.05 or w >= 1.95:
            continue
        h = 2.0 - w
        seed = _grid_seed(w, h)
        cons = _make_constraints(w, h)
        bounds = []
        for i in range(N):
            bounds.append((1e-4, min(w, h) / 2.0))
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))

        try:
            res = minimize(
                lambda x: _objective(x, w),
                seed,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-10},
            )
        except Exception:
            continue

        x = res.x
        # Project radii to be positive
        x[:N] = np.maximum(x[:N], 1e-4)
        if _is_feasible(x, w):
            val = _sum_radii(x)
            if val > best_val:
                best_val = val
                best_result = (x, w)

        if time.time() - _START > _DEADLINE - 10:
            break

    # If nothing feasible, return a trivial valid packing
    if best_result is None:
        w = 1.0
        h = 1.0
        r = 1e-4
        x = np.zeros(3 * N)
        x[:N] = r
        # spread centers
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = ys
        return np.column_stack((xs, ys, r))

    x, w = best_result
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    # Clamp to be safe
    r = np.maximum(r, 1e-4)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)
    # Re-check feasibility after clamping; if broken, shrink radii slightly
    if not _is_feasible(np.concatenate([r, cx, cy]), w, tol=1e-5):
        scale = 0.999
        r = r * scale
        # re-clamp
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)

    return np.column_stack((cx, cy, r))
