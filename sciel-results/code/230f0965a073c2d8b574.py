"""Circle packing in perimeter-4 rectangle: SLSQP with analytic gradient + multi-start hex init."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 138.0  # safety margin before 150s kill

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _objective_penalty(x, W, H, mu=1e3):
    """Return (value, grad) where value = -sum(r) + mu * penalty.
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
    # violation_b[i] = max(r[i] - lo[i], 0)
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
        # d/dx of (r-lo)^2 = 2*(r-lo)*(-d(lo)/dx) for x part
        # d/dx of (r-lo)^2 = 2*(r-lo)*(-1) if left is active (lo = cx, d(lo)/dx = 1)
        # Actually: d/dcx (r - lo)^2 = 2*(r-lo)*(-d(lo)/dcx)
        # If lo = cx, d(lo)/dcx = 1, so term = -2*(r-lo)
        # If lo = W-cx, d(lo)/dcx = -1, so term = +2*(r-lo)
        # Our g_cx above: +1 for left, -1 for right. This is d(lo)/dcx.
        # So grad_cx += 2*mu*viol_b[i] * (-g_cx)
        grad[i, 0] += 2.0 * mu * viol_b[i] * (-g_cx)
        grad[i, 1] += 2.0 * mu * viol_b[i] * (-g_cy)
        # d/dr of (r-lo)^2 = 2*(r-lo)
        grad[i, 2] += 2.0 * mu * viol_b[i]

    # Overlap: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(max(dist_sq, 1e-12))
            rad_sum = r[i] + r[j]
            # violation = max(rad_sum - dist, 0)
            viol = max(rad_sum - dist, 0.0)
            if viol > 0:
                val += mu * viol ** 2
                # grad of (rad_sum - dist)^2 w.r.t cx[i]
                # d/dcx[i] (rad_sum - dist)^2 = 2*(rad_sum - dist) * (-(d(dist)/dcx[i]))
                # d(dist)/dcx[i] = dx/dist
                # So grad_cx[i] += 2*mu*viol * (-dx/dist)
                # grad_cx[j] += 2*mu*viol * (dx/dist)
                # grad_r[i] += 2*mu*viol * 1
                # grad_r[j] += 2*mu*viol * 1
                factor = 2.0 * mu * viol / dist
                grad[i, 0] += -factor * dx
                grad[j, 0] += factor * dx
                grad[i, 1] += -factor * dy
                grad[j, 1] += factor * dy
                grad[i, 2] += 2.0 * mu * viol
                grad[j, 2] += 2.0 * mu * viol

    return val, grad

def _hex_init(W, H, seed=42):
    """Generate a hexagonal packing initialization."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Estimate radius for hex packing
    # For N circles in W x H, density ~ pi/6 * (2r)^2 * (1/area)
    # Let's just use a grid-like hex start
    n = N
    aspect = W / H
    if aspect >= 1:
        n_cols = int(np.round(np.sqrt(n * aspect)))
        n_rows = int(np.ceil(n / n_cols))
    else:
        n_rows = int(np.round(np.sqrt(n / aspect)))
        n_cols = int(np.ceil(n / n_rows))
    
    while n_cols * n_rows < n:
        if aspect >= 1:
            n_cols += 1
        else:
            n_rows += 1
    
    cell_w = W / n_cols
    cell_h = H / n_rows
    base_r = min(cell_w, cell_h) * 0.4
    
    pts = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(pts) >= n:
                break
            # Hex offset
            offset = 0.0
            if i % 2 == 1:
                offset = 0.5 * cell_w
            x = (j + 0.5 + offset) * cell_w
            y = (i + 0.5) * cell_h
            # Clamp to bounds
            x = np.clip(x, base_r, W - base_r)
            y = np.clip(y, base_r, H - base_r)
            pts.append((x, y, base_r))
    
    # Fill remaining if any
    while len(pts) < n:
        x = random.uniform(base_r, W - base_r)
        y = random.uniform(base_r, H - base_r)
        pts.append((x, y, base_r))
    
    pts = pts[:n]
    x = np.array([(p[0] + random.uniform(-0.01, 0.01), p[1] + random.uniform(-0.01, 0.01), p[2]) for p in pts])
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    best_result = None
    best_score = -np.inf
    
    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_c, y_c, r_c])
                return True
        return False

    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.5, 0.6]
    random.seed(42)
    np.random.seed(42)
    random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 - W
        
        x0 = _hex_init(W, H, seed=random.randint(0, 10000))
        
        # SLSQP optimization
        try:
            res = minimize(
                lambda x: _objective_penalty(x, W, H, mu=1e3)[0],
                x0,
                jac=lambda x: _objective_penalty(x, W, H, mu=1e3)[1],
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < _objective_penalty(x0, W, H, mu=1e3)[0]:
                x_opt = res.x
                cx, cy, r = x_opt[:, 0], x_opt[:, 1], x_opt[:, 2]
                update_best(W, H, cx, cy, r)
        except Exception:
            pass
        
        # If time remains, try a second refinement with higher mu
        if time.time() < deadline - 5.0 and best_result is not None:
            try:
                res2 = minimize(
                    lambda x: _objective_penalty(x, W, H, mu=1e4)[0],
                    res.x if 'res' in locals() and res.success else x0,
                    jac=lambda x: _objective_penalty(x, W, H, mu=1e4)[1],
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if res2.success:
                    x_opt2 = res2.x
                    cx2, cy2, r2 = x_opt2[:, 0], x_opt2[:, 1], x_opt2[:, 2]
                    update_best(W, H, cx2, cy2, r2)
            except Exception:
                pass

    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
