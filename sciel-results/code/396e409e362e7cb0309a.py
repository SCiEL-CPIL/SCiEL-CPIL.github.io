import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # Safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    # Check boundary
    for i in range(n):
        if cx[i] < r[i] - 1e-7 or cx[i] > W - r[i] + 1e-7:
            return None
        if cy[i] < r[i] - 1e-7 or cy[i] > H - r[i] + 1e-7:
            return None
        if r[i] < 1e-6:
            return None
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _objective_penalty(x, W, H, mu=500.0):
    """Return (value, grad) for penalty objective."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    # Clamp for numerical stability
    cx = np.clip(cx, -0.5, W + 0.5)
    cy = np.clip(cy, -0.5, H + 0.5)
    r = np.clip(r, -0.5, 2.0)

    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary penalty: r <= min(cx, cy, W-cx, H-cy)
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
        eps = 1e-9
        g_cx = 0.0
        g_cy = 0.0
        cnt = 0
        if d_left <= min_d + eps:
            g_cx += 1.0; cnt += 1
        if d_right <= min_d + eps:
            g_cx += -1.0; cnt += 1
        if d_bottom <= min_d + eps:
            g_cy += 1.0; cnt += 1
        if d_top <= min_d + eps:
            g_cy += -1.0; cnt += 1
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    # r > 0 penalty
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            grad[i, 2] += 2.0 * mu * viol_r[i] * (-1.0)

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

    # Non-overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * (-1.0)
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            dv = 2.0 * mu * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _grid_layout(kx, ky, W, H, shrink=0.99):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    if kx * ky < N:
        # Need to adjust
        while kx * ky < N:
            if kx <= ky:
                kx += 1
            else:
                ky += 1
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
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _optimize_from(x0, W, H, max_iter=150):
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
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Second pass with tighter tolerance if time allows
    if time.time() < DEADLINE - 3:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 100, 'ftol': 1e-12, 'disp': 0})
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
    best_score = -np.inf
    best_result = None

    # Try different aspect ratios and initializations
    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1, 0.85, 1.15]
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0.3:
            continue
        
        # Grid initialization with different grid shapes
        grid_configs = [
            (4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)
        ]
        
        for kx, ky in grid_configs:
            if time.time() > DEADLINE - 10:
                break
            
            x0 = _grid_layout(kx, ky, W, H, shrink=0.95)
            # Add small noise
            x0[:N*2] += np.random.normal(0, 0.02, N*2)
            
            x_opt, score = _optimize_from(x0, W, H, max_iter=120)
            
            if score > best_score:
                best_score = score
                best_result = x_opt
        
        # Random initialization
        if time.time() > DEADLINE - 10:
            break
        
        x0 = _random_layout(W, H)
        x_opt, score = _optimize_from(x0, W, H, max_iter=120)
        
        if score > best_score:
            best_score = score
            best_result = x_opt
    
    if best_result is None:
        # Fallback: simple grid in 1x1 square
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
