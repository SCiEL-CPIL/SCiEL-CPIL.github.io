"""Circle packing in perimeter-4 rectangle: Penalty-based L-BFGS-B with hexagonal seeds."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_START = None
_DEADLINE = 145.0  # 5s safety margin from 150s limit


def _get_W_H(W):
    H = 2.0 - W
    return W, H


def _calc_penalty_and_obj(x, W_fixed=None):
    """
    x layout: [cx_0..cx_{N-1}, cy_0..cy_{N-1}, r_0..r_{N-1}]
    W is fixed externally to reduce search space dimensionality.
    """
    if W_fixed is None:
        W = x[0]
        H = 2.0 - W
        cx = x[1:N+1]
        cy = x[N+1:2*N+1]
        r = x[2*N+1:3*N+1]
    else:
        W = W_fixed
        H = 2.0 - W
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
    
    if W <= 0.1 or H <= 0.1:
        return 1e6, np.zeros_like(x)

    # Objective: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    
    # Gradient for radii (from objective)
    if W_fixed is None:
        grad[2*N+1:3*N+1] = -1.0
    else:
        grad[2*N:3*N] = -1.0

    # Penalty weight
    mu = 1000.0
    eps = 1e-4

    # Wall constraints
    d_left = cx
    d_bottom = cy
    d_right = W - cx
    d_top = H - cy
    
    d_min = np.minimum(np.minimum(d_left, d_bottom), np.minimum(d_right, d_top))
    viol_wall = r - d_min
    
    # Smooth penalty
    viol_smooth = 0.5 * (viol_wall + np.sqrt(viol_wall**2 + eps**2))
    pen_wall = np.sum(viol_smooth**2)
    obj += mu * pen_wall
    
    grad_viol = 0.5 * (1.0 + viol_wall / np.sqrt(viol_wall**2 + eps**2))
    d_pen_wall = 2.0 * viol_smooth * grad_viol

    for i in range(N):
        if d_pen_wall[i] == 0:
            continue
        
        min_d = d_min[i]
        g_cx = 0.0
        g_cy = 0.0
        cnt = 0
        if d_left[i] <= min_d + 1e-9:
            g_cx += 1.0
            cnt += 1
        if d_right[i] <= min_d + 1e-9:
            g_cx += -1.0
            cnt += 1
        if d_bottom[i] <= min_d + 1e-9:
            g_cy += 1.0
            cnt += 1
        if d_top[i] <= min_d + 1e-9:
            g_cy += -1.0
            cnt += 1
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
            
        if W_fixed is None:
            grad[1+i] -= mu * d_pen_wall[i] * g_cx
            grad[N+1+i] -= mu * d_pen_wall[i] * g_cy
            grad[2*N+1+i] += mu * d_pen_wall[i]
        else:
            grad[i] -= mu * d_pen_wall[i] * g_cx
            grad[N+i] -= mu * d_pen_wall[i] * g_cy
            grad[2*N+i] += mu * d_pen_wall[i]

    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            
            viol_smooth = 0.5 * (viol + np.sqrt(viol**2 + eps**2))
            pen = viol_smooth**2
            obj += mu * pen
            grad_viol = 0.5 * (1.0 + viol / np.sqrt(viol**2 + eps**2))
            d_pen = 2.0 * viol_smooth * grad_viol
            
            if d < 1e-12:
                if W_fixed is None:
                    grad[1+i] += mu * d_pen * 1.0
                    grad[1+j] += mu * d_pen * -1.0
                    grad[2*N+1+i] += mu * d_pen * 1.0
                    grad[2*N+1+j] += mu * d_pen * 1.0
                else:
                    grad[i] += mu * d_pen * 1.0
                    grad[j] += mu * d_pen * -1.0
                    grad[2*N+i] += mu * d_pen * 1.0
                    grad[2*N+j] += mu * d_pen * 1.0
            else:
                if W_fixed is None:
                    grad[1+i] += mu * d_pen * (-dx / d)
                    grad[N+1+i] += mu * d_pen * (-dy / d)
                    grad[1+j] += mu * d_pen * (dx / d)
                    grad[N+1+j] += mu * d_pen * (dy / d)
                    grad[2*N+1+i] += mu * d_pen * 1.0
                    grad[2*N+1+j] += mu * d_pen * 1.0
                else:
                    grad[i] += mu * d_pen * (-dx / d)
                    grad[N+i] += mu * d_pen * (-dy / d)
                    grad[j] += mu * d_pen * (dx / d)
                    grad[N+j] += mu * d_pen * (dy / d)
                    grad[2*N+i] += mu * d_pen * 1.0
                    grad[2*N+j] += mu * d_pen * 1.0
                    
    return obj, grad


def _evaluate(x, W):
    """
    Evaluate if x is valid. Return sum of radii or None.
    x layout: [cx_0..cx_{N-1}, cy_0..cy_{N-1}, r_0..r_{N-1}]
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return None
        
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    # Check walls
    for i in range(N):
        if r[i] > min(cx[i], cy[i], W-cx[i], H-cy[i]) + 1e-5:
            return None
            
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            if r[i] + r[j] > d + 1e-5:
                return None
                
    return float(np.sum(r))


def _hex_init(W, H):
    """
    Create a hexagonal grid initialization for 21 circles.
    """
    # Try different grid configurations
    best_x = None
    best_score = -1e9
    
    for rows in range(3, 7):
        cols = (N + rows - 1) // rows
        if cols < 1:
            continue
            
        # Calculate spacing
        # Hexagonal packing: horizontal spacing = 2*r, vertical = sqrt(3)*r
        # We want to fit cols * rows circles.
        # Let's just use a rectangular grid with jitter for simplicity and robustness,
        # but structured to be close to hexagonal.
        
        # Actually, let's stick to a simple grid seed that is feasible, then let L-BFGS-B do the work.
        # The penalty method is good at escaping local minima if the seed is reasonable.
        
        cw = W / cols
        ch = H / rows
        r = 0.5 * min(cw, ch) * 0.9
        
        cx = np.zeros(N)
        cy = np.zeros(N)
        r_arr = np.zeros(N)
        
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx[idx] = (i + 0.5) * cw
                cy[idx] = (j + 0.5) * ch
                r_arr[idx] = r
                idx += 1
        
        # Check feasibility
        if _evaluate(np.concatenate([cx, cy, r_arr]), W) is not None:
            score = np.sum(r_arr)
            if score > best_score:
                best_score = score
                best_x = np.concatenate([cx, cy, r_arr])
                
    if best_x is None:
        # Fallback: tiny circles
        cx = np.linspace(W/(N+1), W - W/(N+1), N)
        cy = np.full(N, H/2.0)
        r_arr = np.full(N, 1e-4)
        best_x = np.concatenate([cx, cy, r_arr])
        
    return best_x


def construct():
    global _START
    if _START is None:
        _START = time.time()

    best_result = None
    best_val = -1e9

    # Try a few aspect ratios
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7]
    
    for w in aspect_ratios:
        if time.time() - _START > _DEADLINE - 10:
            break
        if w <= 0.05 or w >= 1.95:
            continue
        h = 2.0 - w
        
        seed = _hex_init(w, h)
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        for i in range(N):
            bounds.append((1e-4, min(w, h) / 2.0))

        try:
            res = minimize(
                lambda x: _calc_penalty_and_obj(x, W_fixed=w)[0],
                seed,
                jac=lambda x: _calc_penalty_and_obj(x, W_fixed=w)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False},
            )
        except Exception:
            continue

        x = res.x
        # Project radii to be positive
        x[2*N:3*N] = np.maximum(x[2*N:3*N], 1e-4)
        
        val = _evaluate(x, w)
        if val is not None:
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
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        return np.column_stack((xs, ys, r))

    x, w = best_result
    h = 2.0 - w
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    # Clamp to be safe
    r = np.maximum(r, 1e-4)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)
    
    # Re-check feasibility after clamping; if broken, shrink radii slightly
    if _evaluate(np.concatenate([cx, cy, r]), w) is None:
        scale = 0.999
        r = r * scale
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)
        
    return np.column_stack((cx, cy, r))
