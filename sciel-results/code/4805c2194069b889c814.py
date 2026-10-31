"""
Circle packing in a perimeter-4 rectangle.
Combined approach: analytic-gradient SLSQP with multi-start from diverse
initializations, plus a final polishing pass.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 138.0  # Safety margin before 150s kill


def _rect_from_width(W):
    H = 2.0 - W
    return W, H


def _objective_penalty(x, W, H, mu=1e3):
    """
    Return (value, grad) where value = -sum(r) + mu * penalty.
    x is shape (N,3): cx, cy, r.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    # Clamp to a reasonable range for numerical stability
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    # Objective: maximize sum(r) -> minimize -sum(r)
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary: r <= min(cx, cy, W-cx, H-cy), r > 0
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)

    # grad for boundary violation
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        if d_left <= min_d + eps:
            g_cx += 1.0
        if d_right <= min_d + eps:
            g_cx += -1.0
        if d_bottom <= min_d + eps:
            g_cy += 1.0
        if d_top <= min_d + eps:
            g_cy += -1.0
        cnt = (d_left <= min_d + eps) + (d_right <= min_d + eps) + \
              (d_bottom <= min_d + eps) + (d_top <= min_d + eps)
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    # r > 0: violation if r < 0
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            dv = 2.0 * mu * viol_r[i]
            grad[i, 2] += dv * (-1.0)

    # Centers in box (soft)
    mu_c = mu * 0.1
    viol_cx_lo = np.maximum(-cx, 0.0)
    viol_cx_hi = np.maximum(cx - W, 0.0)
    viol_cy_lo = np.maximum(-cy, 0.0)
    viol_cy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(viol_cx_lo**2) + np.sum(viol_cx_hi**2) +
                   np.sum(viol_cy_lo**2) + np.sum(viol_cy_hi**2))
    for i in range(n):
        if viol_cx_lo[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_lo[i] * (-1.0)
        if viol_cx_hi[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_hi[i] * 1.0
        if viol_cy_lo[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_lo[i] * (-1.0)
        if viol_cy_hi[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_hi[i] * 1.0

    # Non-overlap: for each pair, max(r_i + r_j - dist, 0)^2
    mu_o = mu
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu_o * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu_o * viol * 1.0
                grad[j, 0] += 2.0 * mu_o * viol * (-1.0)
                grad[i, 2] += 2.0 * mu_o * viol * 1.0
                grad[j, 2] += 2.0 * mu_o * viol * 1.0
                continue
            dv = 2.0 * mu_o * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad


def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _grid_layout(kx, ky, W, H, shrink=0.999):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _hexagonal_layout(W, H):
    """Hexagonal close-packed layout."""
    pts = []
    r_est = min(W, H) / 6.0
    dx = 2.0 * r_est * 0.999
    dy = dx * np.sqrt(3) / 2.0
    row = 0
    y = r_est
    while y < H - r_est * 0.5 and len(pts) < N:
        offset = 0.0 if row % 2 == 0 else dx / 2.0
        x = r_est + offset
        while x < W - r_est * 0.5 and len(pts) < N:
            r = min(x, y, W - x, H - y) * 0.999
            r = max(r, 1e-4)
            pts.append([x, y, r])
            x += dx
        y += dy
        row += 1
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


def _optimize_from(x0, W, H, max_iter=200, tight_ftol=1e-10):
    """Run SLSQP from x0. Returns (best_x, best_sum)."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    def fun(x):
        v, _ = _objective_penalty(x, W, H)
        return v

    def grad(x):
        _, g = _objective_penalty(x, W, H)
        return g

    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': tight_ftol,
                                'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Second pass from best_x with tighter tolerance
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 150, 'ftol': 1e-12,
                                     'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum


def construct():
    start_time = time.time()
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() - start_time < DEADLINE - 5

    # Try multiple aspect ratios and initializations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.9, 1.1]
    
    for W in aspect_ratios:
        if not time_ok():
            break
        H = 2.0 - W
        if W < 0.3 or H < 0.3:
            continue

        # Multiple initializations
        inits = []
        # Grid layouts
        inits.append(_grid_layout(5, 5, W, H))
        inits.append(_grid_layout(4, 6, W, H))
        inits.append(_grid_layout(6, 4, W, H))
        inits.append(_grid_layout(3, 7, W, H))
        inits.append(_grid_layout(7, 3, W, H))
        # Hexagonal
        inits.append(_hexagonal_layout(W, H))
        # Random
        for _ in range(3):
            inits.append(_random_layout(W, H))

        for x0 in inits:
            if not time_ok():
                break
            x_opt, s = _optimize_from(x0, W, H, max_iter=200)
            if s > best_sum:
                best_sum = s
                best_arr = x_opt.copy()
                best_W = W
                best_H = H

    # Final polishing: take the best found and run a few more SLSQP passes
    if best_arr is not None and time_ok():
        W, H = best_W, best_H
        for _ in range(3):
            if not time_ok():
                break
            x_opt, s = _optimize_from(best_arr, W, H, max_iter=100, tight_ftol=1e-12)
            if s > best_sum:
                best_sum = s
                best_arr = x_opt.copy()

    # Build result
    if best_arr is not None and best_sum > 0:
        cx = np.clip(best_arr[:, 0], 0.0, best_W)
        cy = np.clip(best_arr[:, 1], 0.0, best_H)
        r = np.maximum(best_arr[:, 2], 1e-6)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
        r = np.minimum(r, lo)
        r = np.maximum(r, 1e-6)
        result = np.column_stack((cx, cy, r))
        return result

    # Absolute fallback
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    return np.array(cells[:N])
