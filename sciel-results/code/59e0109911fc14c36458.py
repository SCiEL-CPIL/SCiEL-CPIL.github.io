"""Circle packing in perimeter-4 rectangle: force-directed + SLSQP hybrid."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 138.0

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _objective_penalty(x, W, H, mu=1e3):
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
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
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            dv = 2.0 * mu * viol_r[i]
            grad[i, 2] += dv * (-1.0)
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

def _random_layout(W, H):
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _force_relax(x, W, H, max_iter=500, growth_rate=0.001, lr=0.05):
    """Force-directed relaxation with radius growth. Returns improved x."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        # Try to grow radii
        r_new = r * (1.0 + growth_rate)
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                min_dist = r_new[i] + r_new[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-8:
                        nx = dx / dist
                        ny = dy / dist
                        fx[i] += nx * overlap
                        fy[i] += ny * overlap
                        fx[j] -= nx * overlap
                        fy[j] -= ny * overlap
                    else:
                        angle = random.uniform(0, 2 * np.pi)
                        fx[i] += np.cos(angle) * overlap
                        fy[i] += np.sin(angle) * overlap
                        fx[j] -= np.cos(angle) * overlap
                        fy[j] -= np.sin(angle) * overlap
        
        # Wall constraints
        for i in range(n):
            if cx[i] < r_new[i]:
                fx[i] += (r_new[i] - cx[i]) * 5.0
            if cx[i] > W - r_new[i]:
                fx[i] -= (cx[i] - (W - r_new[i])) * 5.0
            if cy[i] < r_new[i]:
                fy[i] += (r_new[i] - cy[i]) * 5.0
            if cy[i] > H - r_new[i]:
                fy[i] -= (cy[i] - (H - r_new[i])) * 5.0
        
        # Update positions
        cx_new = cx + lr * fx
        cy_new = cy + lr * fy
        
        # Project to bounds
        cx_new = np.clip(cx_new, r_new, W - r_new)
        cy_new = np.clip(cy_new, r_new, H - r_new)
        
        # Check validity
        valid = True
        for i in range(n):
            if cx_new[i] < r_new[i] - 1e-8 or cx_new[i] > W - r_new[i] + 1e-8:
                valid = False
                break
            if cy_new[i] < r_new[i] - 1e-8 or cy_new[i] > H - r_new[i] + 1e-8:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx_new[i] - cx_new[j]
                    dy = cy_new[i] - cy_new[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = r_new[i] + r_new[j]
                    if dist_sq < min_dist * min_dist - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            cx = cx_new
            cy = cy_new
            r = r_new
        else:
            growth_rate *= 0.9
            if growth_rate < 1e-6:
                break
    
    return np.column_stack([cx, cy, r])

def _optimize_from(x0, W, H, max_iter=200):
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    # First: force relaxation to improve starting point
    x_relaxed = _force_relax(x0, W, H, max_iter=300)
    s_relaxed = _evaluate(x_relaxed, W, H)
    if s_relaxed is not None and s_relaxed > best_sum:
        best_sum = s_relaxed
        best_x = x_relaxed

    def fun(x):
        v, _ = _objective_penalty(x, W, H)
        return v

    def grad(x):
        _, g = _objective_penalty(x, W, H)
        return g

    try:
        res = minimize(fun, best_x, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10,
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
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    candidates = []

    Ws = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5)]

    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.99)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    for W in Ws:
        H = 2.0 - W
        for _ in range(3):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    candidates.sort(key=lambda t: -t[0])

    top_k = min(12, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        remaining = DEADLINE - time.time()
        max_iter = 150 if remaining > 60 else 80
        x_opt, s_opt = _optimize_from(g0, W, H, max_iter=max_iter)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(8, int(remaining // 8))
        for _ in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0 = np.clip(x0, 0.0, None)
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _optimize_from(x0, best_W, best_H, max_iter=60)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.99)
        best_arr = g
        best_W, best_H = W, H

    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in candidates[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = np.maximum(g0[:, 2], 1e-6)
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.minimum(r2, lo2)
            r2 = np.maximum(r2, 1e-6)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
