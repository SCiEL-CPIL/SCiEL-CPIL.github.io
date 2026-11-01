import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 138.0  # evaluator kills at 150s; keep 12s margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _objective(params, start, soft_eps=1e-4, mu=50.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start, soft_eps=1e-6, mu=200.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_grad(params, start):
    """Objective with analytic gradient for L-BFGS-B."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros(3*N)
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:3*N].copy()
    r_safe = np.maximum(r, 1e-10)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    dist[dist < 1e-14] = 1e-14
    min_dist = r_safe[:, None] + r_safe[None, :]
    diff = min_dist - dist
    
    eps = 1e-8
    sqrt_term = np.sqrt(diff*diff + eps)
    sqrt_term[sqrt_term < 1e-14] = 1e-14
    dsmooth = 0.5 * (1 + diff / sqrt_term)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    obj = -np.sum(r_safe)
    overlap_pen = np.sum(dsmooth * mask)
    
    dneg_x = r_safe - x
    dpos_x = x - (1.0 - r_safe)
    dneg_y = r_safe - y
    dpos_y = y - (1.0 - r_safe)
    
    sqrt_dx = np.sqrt(dneg_x*dneg_x + eps)
    sqrt_dx[sqrt_dx < 1e-14] = 1e-14
    sm_dn_x = 0.5 * (1 + dneg_x / sqrt_dx)
    
    sqrt_px = np.sqrt(dpos_x*dpos_x + eps)
    sqrt_px[sqrt_px < 1e-14] = 1e-14
    sm_dp_x = 0.5 * (1 + dpos_x / sqrt_px)
    
    sqrt_dy = np.sqrt(dneg_y*dneg_y + eps)
    sqrt_dy[sqrt_dy < 1e-14] = 1e-14
    sm_dn_y = 0.5 * (1 + dneg_y / sqrt_dy)
    
    sqrt_py = np.sqrt(dpos_y*dpos_y + eps)
    sqrt_py[sqrt_py < 1e-14] = 1e-14
    sm_dp_y = 0.5 * (1 + dpos_y / sqrt_py)
    
    bound_pen = np.sum(0.5*(dneg_x + sqrt_dx) + 0.5*(dpos_x + sqrt_px) + 
                       0.5*(dneg_y + sqrt_dy) + 0.5*(dpos_y + sqrt_py))
    
    mu = 50.0
    obj_val = obj + mu * (overlap_pen + bound_pen)
    
    grad = np.zeros(3*N)
    
    # Gradient w.r.t. x
    for i in range(N):
        for j in range(i+1, N):
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                # d(overlap_pen)/dx[i]
                # overlap_pen includes dsmooth[i,j]
                # dsmooth = 0.5*(1 + diff/sqrt(diff^2+eps))
                # diff = r_i + r_j - d
                # d(diff)/dx[i] = -d(d)/dx[i] = -ux (since d = sqrt((x_i-x_j)^2+...), d(d)/dx[i] = (x_i-x_j)/d = ux)
                # Wait, d = dist[i,j]. d(dist[i,j])/dx[i] = (x[i]-x[j])/dist[i,j] = ux
                # diff = r_i + r_j - dist[i,j]
                # d(diff)/dx[i] = -ux
                # dsmooth/d(diff) = 0.5 / sqrt(diff^2+eps) = 0.5/sqrt_term[i,j]
                # d(overlap_pen)/dx[i] = mu * dsmooth[i,j]'s derivative? No, overlap_pen = sum(dsmooth[mask])
                # d(overlap_pen)/dx[i] = mu * d(dsmooth[i,j])/dx[i] = mu * (dsmooth/ddiff) * (ddiff/dx[i])
                # = mu * (0.5 / sqrt_term[i,j]) * (-ux)
                
                # But we also have the term -np.sum(r) which doesn't depend on x.
                # And bound_pen doesn't depend on x directly? Wait, bound_pen depends on x.
                # bound_pen = sum(sm(r-x) + sm(x-(1-r)) + ...)
                # d(sm(r-x))/dx[i] = sm'(r-x) * (-1) = -sm_dn_x[i]
                # d(sm(x-(1-r)))/dx[i] = sm'(x-(1-r)) * (1) = sm_dp_x[i]
                
                # So grad_x[i] = mu * (-0.5 / sqrt_term[i,j] * ux) + mu * (-sm_dn_x[i] + sm_dp_x[i])
                # Wait, the sum over j for i is in the loop.
                pass
        
        # It's easier to compute the full gradient matrix and sum.
        # grad_x[i] = -mu * sum_j dsmooth[i,j] * (d(dist[i,j])/dx[i]) + mu * (-sm_dn_x[i] + sm_dp_x[i])
        # Note: dsmooth is symmetric, mask is upper triangle.
        # For i, j > i: term is -mu * dsmooth[i,j] * ux_ij
        # For j, i > j: term is -mu * dsmooth[j,i] * ux_ji = -mu * dsmooth[i,j] * (-ux_ij) = mu * dsmooth[i,j] * ux_ij
        # So grad_x[i] = -mu * sum_{j>i} dsmooth[i,j]*ux_ij + mu * sum_{j<i} dsmooth[j,i]*ux_ji
        # Since dsmooth[j,i]=dsmooth[i,j] and ux_ji = -ux_ij, this is:
        # grad_x[i] = -mu * sum_{j>i} dsmooth[i,j]*ux_ij - mu * sum_{j<i} dsmooth[i,j]*ux_ij
        # = -mu * sum_{j!=i} dsmooth[i,j]*ux_ij (where dsmooth[i,i]=0)
        
        # Let's just accumulate.
        for j in range(N):
            if i == j: continue
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                # dsmooth[i,j] is the derivative of the smooth max term for pair (i,j)
                # In the objective, we sum dsmooth[mask].
                # dsmooth is the derivative of the smooth max function with respect to its argument (diff).
                # So the contribution to grad_x[i] from pair (i,j) is:
                # mu * (dsmooth[i,j]) * (d(diff[i,j])/dx[i])
                # d(diff[i,j])/dx[i] = -ux
                # So contribution is -mu * dsmooth[i,j] * ux
                # But wait, dsmooth[i,j] in my code above is 0.5*(1+diff/sqrt(...)).
                # This is the derivative of the smooth max function.
                # So yes, grad_x[i] += -mu * dsmooth[i,j] * ux
                grad[i] += -mu * dsmooth[i, j] * ux
                grad[N+i] += -mu * dsmooth[i, j] * uy
                
                # Gradient w.r.t. r[i]
                # obj = -sum(r) -> grad_r[i] += -1
                # overlap_pen: dsmooth[i,j] * d(diff[i,j])/dr[i] = dsmooth[i,j] * 1
                # So grad_r[i] += mu * dsmooth[i,j]
                grad[2*N+i] += mu * dsmooth[i, j]
                
                # bound_pen: d(sm(r-x))/dr[i] = sm'(r-x)*1 = sm_dn_x[i]
                # d(sm(x-(1-r)))/dr[i] = sm'(x-(1-r))*(-1) = -sm_dp_x[i]
                # d(sm(r-y))/dr[i] = sm_dn_y[i]
                # d(sm(y-(1-r)))/dr[i] = -sm_dp_y[i]
                # These don't depend on j, so they should be added once per i, not per j.
                # I'll add them after the loop.
                
    # Add boundary gradients and objective gradient
    for i in range(N):
        grad[i] += mu * (-sm_dn_x[i] + sm_dp_x[i])
        grad[N+i] += mu * (-sm_dn_y[i] + sm_dp_y[i])
        grad[2*N+i] += -1.0  # from -sum(r)
        grad[2*N+i] += mu * (sm_dn_x[i] - sm_dp_x[i] + sm_dn_y[i] - sm_dp_y[i])
        
    return obj_val, grad

def _push_apart(x, y, r, iters=100):
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / d, dy / d
                    gap = need_ij - d
                    step = 0.5 * gap
                    x[i] += step * ux
                    y[i] += step * uy
                    x[j] -= step * ux
                    y[j] -= step * uy
                    moved = True
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if not moved:
            break
    return x, y

def _make_valid(x, y, r):
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(50):
        x, y = _push_apart(x, y, r, iters=100)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if _is_valid(x, y, r):
            break
    if not _is_valid(x, y, r):
        for _ in range(500):
            for i in range(N):
                mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > mr * 0.999:
                    r[i] = max(0.0, mr * 0.998)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need - 1e-10
            if not np.any(overlaps):
                break
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i: i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
    return x, y, r

def _hex_init():
    rows = [6, 5, 6, 5, 6, 4]
    x, y, r = [], [], []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            cr = 0.48 * row_h
            cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
            x.append(cx); y.append(ry); r.append(cr)
    x = np.array(x[:N]); y = np.array(y[:N]); r = np.array(r[:N])
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.07)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # 1. L-BFGS-B with gradient
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            res = minimize(
                _objective_grad, params, args=(start,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # 2. Powell polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                lambda p: _objective(p, start), p0,
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
            
    return best

def _final_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Tight L-BFGS-B
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                lambda p: _objective_tight(p, start), p0,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Local perturbation
    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(123)
        for _ in range(30):
            if _elapsed(start) > TIME_LIMIT - 1: break
            scale = 0.01 * (1 - _ / 30.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = best[2] + rng.normal(0, scale * 0.5, N)
            pr = np.maximum(pr, 1e-4)
            vx, vy, vr = _make_valid(px, py, pr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
    
    return best

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(8, 4))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 10: break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    if _elapsed(start) < TIME_LIMIT - 5:
        res = _final_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    return np.column_stack((x, y, r))
