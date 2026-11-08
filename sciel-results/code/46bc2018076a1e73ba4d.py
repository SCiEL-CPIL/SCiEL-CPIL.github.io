"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 135.0  # safety margin vs 150s kill


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


def _make_grid(cols, rows, W, H, n, jitter=0.0, seed=0):
    """Place n circles in a cols x rows grid within W x H, with optional jitter."""
    xs = np.zeros(n); ys = np.zeros(n); rs = np.zeros(n)
    cw = W / cols
    ch = H / rows
    for k in range(n):
        ci = k % cols
        ri = k // cols
        xs[k] = (ci + 0.5) * cw
        ys[k] = (ri + 0.5) * ch
    r0 = 0.48 * min(cw, ch)
    rs[:] = r0
    if jitter > 0:
        rng = np.random.default_rng(seed)
        xs += rng.normal(0, jitter, n)
        ys += rng.normal(0, jitter, n)
    return xs, ys, rs


def _objective_and_grads(v, W, H):
    """v = [x1..xn, y1..yn, r1..rn]. Returns (negative sum of radii, gradient)."""
    n = N
    xs = v[:n]; ys = v[n:2*n]; rs = v[2*n:3*n]
    neg_sum = -np.sum(rs)
    
    gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)
    
    # Boundary constraints: want x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
    # Use smooth penalty: for constraint x_i - r_i >= 0, penalty = max(0, r_i - x_i)^2
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        m = xs[i] - rs[i]
        if m < 0:
            gx[i] += 2.0 * m
            gr[i] += -2.0 * m
        # Right boundary: W - x_i - r_i >= 0
        m = W - xs[i] - rs[i]
        if m < 0:
            gx[i] += -2.0 * m
            gr[i] += -2.0 * m
        # Bottom boundary: y_i - r_i >= 0
        m = ys[i] - rs[i]
        if m < 0:
            gy[i] += 2.0 * m
            gr[i] += -2.0 * m
        # Top boundary: H - y_i - r_i >= 0
        m = H - ys[i] - rs[i]
        if m < 0:
            gy[i] += -2.0 * m
            gr[i] += -2.0 * m
    
    # Non-overlap: dist_ij >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.hypot(dx, dy)
            m = d - rs[i] - rs[j]
            if m < 0:
                if d > 1e-9:
                    ux = dx / d; uy = dy / d
                else:
                    ux = 1.0; uy = 0.0
                gx[i] += 2.0 * m * ux
                gx[j] += -2.0 * m * ux
                gy[i] += 2.0 * m * uy
                gy[j] += -2.0 * m * uy
                gr[i] += -2.0 * m
                gr[j] += -2.0 * m
    
    # Total gradient of penalty
    g = np.concatenate([gx, gy, gr])
    # Objective gradient: d(-sum(rs))/dv = [0,0,1,1,...,1] for radii part
    obj_grad = np.zeros(3*n)
    obj_grad[2*n:3*n] = -1.0
    total_grad = obj_grad + g
    return neg_sum, total_grad


def _optimize_scipy(W, H, xs0, ys0, rs0, max_iters=2000, tol=1e-6):
    """Use scipy SLSQP to maximize sum of radii with variable positions and radii."""
    v0 = np.concatenate([xs0, ys0, rs0])
    
    def fun(v):
        val, _ = _objective_and_grads(v, W, H)
        return val
    
    def jac(v):
        _, g = _objective_and_grads(v, W, H)
        return g
    
    # Bounds: 0 <= x_i <= W, 0 <= y_i <= H, 1e-6 <= r_i <= min(W,H)/2
    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    for i in range(N):
        bounds.append((1e-6, 0.5 * min(W, H)))
    
    result = minimize(fun, v0, jac=jac, method='SLSQP', bounds=bounds,
                      options={'maxiter': max_iters, 'ftol': tol})
    
    if result.success or result.fun < fun(v0):
        v_opt = result.x
        xs_opt = v_opt[:N]
        ys_opt = v_opt[N:2*N]
        rs_opt = v_opt[2*N:3*N]
        return xs_opt, ys_opt, rs_opt, result.fun
    return xs0, ys0, rs0, fun(v0)


def _try_layout(cols, rows, aspect, seed=0, max_iters=1500):
    """Try a specific layout with scipy optimization."""
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    
    xs, ys, rs = _make_grid(cols, rows, W, H, N, jitter=0.01, seed=seed)
    
    # Initial penalty-based local search to get close to feasible
    lam = 5.0
    step = 1e-3
    best_val = -np.sum(rs)
    best_v = np.concatenate([xs.copy(), ys.copy(), rs.copy()])
    
    for it in range(500):
        if time.time() > _DEADLINE - 15.0:
            break
        val, g = _objective_and_grads(best_v, W, H)
        # Ascent on sum(r) with penalty
        obj_grad = np.zeros(3*N)
        obj_grad[2*N:3*N] = 1.0  # we want to maximize sum(r), so gradient of sum(r) is +1
        # We're minimizing -sum(r) + penalty, so gradient is -1 for radii + penalty_grad
        # But _objective_and_grads already returns gradient of -sum(r) + penalty
        # So ascent step: v += step * (-g) since g is gradient of what we minimize
        v_new = best_v - step * g
        # Project to bounds
        v_new[:N] = np.clip(v_new[:N], 0, W)
        v_new[N:2*N] = np.clip(v_new[N:2*N], 0, H)
        v_new[2*N:3*N] = np.clip(v_new[2*N:3*N], 1e-6, 0.5*min(W,H))
        
        new_val, _ = _objective_and_grads(v_new, W, H)
        if new_val < best_val:
            best_val = new_val
            best_v = v_new
        else:
            step *= 0.8
        if it % 100 == 0:
            lam *= 1.5  # increase penalty weight implicitly by scaling g? 
            # Actually our g already includes penalty; let's just decay step more
            step *= 0.9
    
    # Now use scipy from this starting point
    xs0 = best_v[:N]; ys0 = best_v[N:2*N]; rs0 = best_v[2*N:3*N]
    
    if time.time() > _DEADLINE - 10.0:
        # Just use the penalty result
        arr = np.column_stack([xs0, ys0, rs0])
        if _validate(arr):
            return arr, np.sum(rs0)
        return None, 0.0
    
    xs_opt, ys_opt, rs_opt, opt_val = _optimize_scipy(W, H, xs0, ys0, rs0, 
                                                       max_iters=max_iters, tol=1e-7)
    
    arr = np.column_stack([xs_opt, ys_opt, rs_opt])
    
    # Validate and fix if needed
    if _validate(arr):
        return arr, np.sum(rs_opt)
    
    # If scipy result is invalid, try to repair
    # Shrink radii slightly
    for factor in [0.99, 0.98, 0.97, 0.95, 0.9]:
        arr2 = arr.copy()
        arr2[:, 2] *= factor
        if _validate(arr2):
            return arr2, np.sum(arr2[:, 2])
    
    # Fallback to penalty result
    arr = np.column_stack([xs0, ys0, rs0])
    if _validate(arr):
        return arr, np.sum(rs0)
    return None, 0.0


def construct():
    best_arr = None
    best_val = -1.0
    
    # Candidate layouts: (cols, rows, aspect)
    # 21 circles: 7x3, 6x4 (24 cells, use 21), 5x5 (25 cells, use 21)
    layouts = [
        (7, 3, 7.0 / 3.0),    # natural aspect for 7x3
        (7, 3, 1.4),
        (7, 3, 1.6),
        (6, 4, 1.5),          # 6x4 grid
        (6, 4, 1.3),
        (5, 5, 1.0),          # square
        (5, 5, 0.9),
        (5, 5, 1.1),
    ]
    
    for idx, (cols, rows, asp) in enumerate(layouts):
        if time.time() > _DEADLINE - 20.0:
            break
        # Try multiple seeds for diversity
        for seed in [0, 1, 2]:
            if time.time() > _DEADLINE - 15.0:
                break
            arr, val = _try_layout(cols, rows, asp, seed=seed, max_iters=1200)
            if arr is not None and val > best_val:
                best_val = val
                best_arr = arr
    
    # Final polish: take best and do one more scipy pass with tighter tolerance
    if best_arr is not None and time.time() < _DEADLINE - 10.0:
        # Compute bounding box of current best
        x = best_arr[:, 0]; y = best_arr[:, 1]; r = best_arr[:, 2]
        xmin, xmax = np.min(x - r), np.max(x + r)
        ymin, ymax = np.min(y - r), np.max(y + r)
        W = xmax - xmin
        H = ymax - ymin
        # Translate to [0,W] x [0,H]
        xs = x - xmin
        ys = y - ymin
        rs = r
        v0 = np.concatenate([xs, ys, rs])
        
        # Do a final polish
        if time.time() < _DEADLINE - 8.0:
            xs_opt, ys_opt, rs_opt, _ = _optimize_scipy(W, H, xs, ys, rs, 
                                                         max_iters=1000, tol=1e-8)
            arr_polish = np.column_stack([xs_opt, ys_opt, rs_opt])
            if _validate(arr_polish):
                val_polish = np.sum(rs_opt)
                if val_polish > best_val:
                    best_val = val_polish
                    best_arr = arr_polish
    
    # Fallback: simple grid in a square
    if best_arr is None or best_val <= 0:
        W = 1.0; H = 1.0
        xs, ys, rs = _make_grid(5, 5, W, H, N, jitter=0)
        # Adjust to fit 21 in 5x5 (skip last 4)
        arr = np.column_stack([xs[:N], ys[:N], rs[:N]])
        best_arr = arr
        best_val = np.sum(arr[:, 2])
    
    return best_arr
