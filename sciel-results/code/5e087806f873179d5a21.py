import numpy as np
from scipy.optimize import minimize, linprog
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

def _make_valid(x, y, r, max_iter=200):
    """Iteratively shrink radii to satisfy all constraints."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            r[i] = min(r[i], x[i] * 0.999, (1.0 - x[i]) * 0.999,
                       y[i] * 0.999, (1.0 - y[i]) * 0.999)
            if r[i] < 1e-6:
                r[i] = 1e-6
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    shrink = (d / min_d) * 0.999 if min_d > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _hex_init():
    """Hexagonal packing initialization for 32 circles."""
    rows = [6, 5, 6, 5, 6, 4]
    x = []
    y = []
    r = []
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
            x.append(cx)
            y.append(ry)
            r.append(cr)
    x = x[:N]
    y = y[:N]
    r = r[:N]
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
        r.append(0.02)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _rect_init():
    """Rectangular grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    x = []
    y = []
    r = []
    for i in range(side):
        for j in range(side):
            if len(x) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                cr = 0.48 / side
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                x.append(cx)
                y.append(cy)
                r.append(cr)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return x, y, r

def _optimize_radii_lp(x, y):
    """
    Solve for maximum radii given fixed centers using Linear Programming.
    Maximize sum(r) subject to r_i + r_j <= d_ij and r_i <= dist_to_bound.
    """
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    for i in range(N):
        b = min(x[i], 1.0-x[i], y[i], 1.0-y[i])
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(b)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    r = np.array([min(x[i], 1.0-x[i], y[i], 1.0-y[i]) for i in range(N)])
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if r[i] + r[j] > d:
                    shrink = d / (r[i] + r[j]) if (r[i] + r[j]) > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed: 
            break
    return r

def _objective(params, soft_eps=1e-4):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    pen = np.zeros(4*N)
    pen[0:N] = r - x
    pen[N:2*N] = x - (1.0 - r)
    pen[2*N:3*N] = r - y
    pen[3*N:4*N] = y - (1.0 - r)
    bound_pen = np.sum(0.5 * (pen + np.sqrt(pen*pen + soft_eps*soft_eps)))
    
    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, soft_eps=1e-6):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    pen = np.zeros(4*N)
    pen[0:N] = r - x
    pen[N:2*N] = x - (1.0 - r)
    pen[2*N:3*N] = r - y
    pen[3*N:4*N] = y - (1.0 - r)
    bound_pen = np.sum(0.5 * (pen + np.sqrt(pen*pen + soft_eps*soft_eps)))
    
    mu = 200.0
    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iterations=50):
    """Physics-based push apart to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    if d < 1e-10:
                        dx, dy = 1e-3, 0.0
                        d = np.hypot(dx, dy)
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        for i in range(N):
            x[i] = np.clip(x[i], r[i]*0.99, 1.0 - r[i]*0.99)
            y[i] = np.clip(y[i], r[i]*0.99, 1.0 - r[i]*0.99)
        if not moved:
            break
    return x, y

def _optimize(x0, start_time, time_budget=100):
    """Run L-BFGS-B optimization."""
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 50000})
        return res.x
    except Exception:
        return x0

def _tight_polish(params, start_time, time_budget=10):
    """Final high-precision polish."""
    try:
        res = minimize(_objective_tight, params, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14,
                                'maxfun': 100000})
        return res.x
    except Exception:
        return params

def _local_search(params, start_time, max_iter=1000):
    """Local search with random perturbations to escape local minima."""
    best_params = params.copy()
    best_score = -_sum_r(best_params[2*N:3*N])
    rng = np.random.RandomState(12345)
    
    for i in range(max_iter):
        if _elapsed(start_time) > TIME_LIMIT - 10:
            break
        
        scale = 0.05 * (1 - i / max_iter)
        perturbed = best_params + rng.normal(0, scale, size=3*N)
        
        px = perturbed[:N].copy()
        py = perturbed[N:2*N].copy()
        pr = perturbed[2*N:3*N].copy()
        px, py = _push_apart(px, py, pr, iterations=20)
        p_opt = np.concatenate([px, py, pr])
        
        try:
            res = minimize(_objective, p_opt, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 5000})
            p_opt = res.x
        except Exception:
            continue
        
        vx = p_opt[:N]
        vy = p_opt[N:2*N]
        vr = p_opt[2*N:3*N]
        vx, vy, vr = _make_valid(vx, vy, vr)
        
        score = _sum_r(vr)
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx, vy, vr])
    
    return best_params

def construct():
    start_time = time.time()
    
    inits = []
    
    hx, hy, hr = _hex_init()
    inits.append(np.concatenate([hx, hy, hr]))
    
    rx, ry, rr = _rect_init()
    inits.append(np.concatenate([rx, ry, rr]))
    
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        rx2, ry2, rr2 = _random_init(seed)
        inits.append(np.concatenate([rx2, ry2, rr2]))
    
    rng = np.random.RandomState(7)
    for _ in range(6):
        hx2, hy2, hr2 = _hex_init()
        hx2 += rng.uniform(-0.05, 0.05, N)
        hy2 += rng.uniform(-0.05, 0.05, N)
        hx2 = np.clip(hx2, 0, 1)
        hy2 = np.clip(hy2, 0, 1)
        inits.append(np.concatenate([hx2, hy2, hr2]))
    
    best_params = None
    best_score = -np.inf
    
    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 30:
            break
        
        x0a = x0[:N].copy()
        x0b = x0[N:2*N].copy()
        x0c = x0[2*N:3*N].copy()
        x0a, x0b = _push_apart(x0a, x0b, x0c, iterations=30)
        x0_opt = np.concatenate([x0a, x0b, x0c])
        
        result = _optimize(x0_opt, start_time)
        
        vx = result[:N]
        vy = result[N:2*N]
        vr = result[2*N:3*N]
        
        # Use LP to find optimal radii for these centers
        vr_lp = _optimize_radii_lp(vx, vy)
        vx, vy, vr_lp = _make_valid(vx, vy, vr_lp)
        
        score_lp = _sum_r(vr_lp)
        
        # Also try the soft-constraint radii
        vx2, vy2, vr2 = _make_valid(vx, vy, vr)
        score_soft = _sum_r(vr2)
        
        if score_lp >= score_soft:
            score = score_lp
            best_candidate = np.concatenate([vx, vy, vr_lp])
        else:
            score = score_soft
            best_candidate = np.concatenate([vx2, vy2, vr2])
        
        if score > best_score:
            best_score = score
            best_params = best_candidate
    
    # Final tight polish on best
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 20:
        polished = _tight_polish(best_params, start_time)
        px = polished[:N]
        py = polished[N:2*N]
        pr = polished[2*N:3*N]
        
        # Use LP for radii after polish
        pr_lp = _optimize_radii_lp(px, py)
        px, py, pr_lp = _make_valid(px, py, pr_lp)
        p_score_lp = _sum_r(pr_lp)
        
        px2, py2, pr2 = _make_valid(px, py, pr)
        p_score_soft = _sum_r(pr2)
        
        if p_score_lp >= p_score_soft:
            p_score = p_score_lp
            p_params = np.concatenate([px, py, pr_lp])
        else:
            p_score = p_score_soft
            p_params = np.concatenate([px2, py2, pr2])
        
        if p_score > best_score:
            best_score = p_score
            best_params = p_params
    
    # Local search from best result
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        ls_result = _local_search(best_params, start_time, max_iter=200)
        ls_x = ls_result[:N]
        ls_y = ls_result[N:2*N]
        ls_r = ls_result[2*N:3*N]
        
        ls_r_lp = _optimize_radii_lp(ls_x, ls_y)
        ls_x, ls_y, ls_r_lp = _make_valid(ls_x, ls_y, ls_r_lp)
        ls_score_lp = _sum_r(ls_r_lp)
        
        ls_x2, ls_y2, ls_r2 = _make_valid(ls_x, ls_y, ls_r)
        ls_score_soft = _sum_r(ls_r2)
        
        if ls_score_lp >= ls_score_soft:
            ls_score = ls_score_lp
            ls_final = np.concatenate([ls_x, ls_y, ls_r_lp])
        else:
            ls_score = ls_score_soft
            ls_final = np.concatenate([ls_x2, ls_y2, ls_r2])
        
        if ls_score > best_score:
            best_score = ls_score
            best_params = ls_final
    
    # One more round of optimization from polished result if time permits
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 10:
        result2 = _optimize(best_params, start_time)
        vx2 = result2[:N]
        vy2 = result2[N:2*N]
        vr2 = result2[2*N:3*N]
        
        vr2_lp = _optimize_radii_lp(vx2, vy2)
        vx2, vy2, vr2_lp = _make_valid(vx2, vy2, vr2_lp)
        s2_lp = _sum_r(vr2_lp)
        
        vx2b, vy2b, vr2b = _make_valid(vx2, vy2, vr2)
        s2_soft = _sum_r(vr2b)
        
        if s2_lp >= s2_soft:
            s2 = s2_lp
            final2 = np.concatenate([vx2, vy2, vr2_lp])
        else:
            s2 = s2_soft
            final2 = np.concatenate([vx2b, vy2b, vr2b])
        
        if s2 > best_score:
            best_score = s2
            best_params = final2
    
    # Final tight polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 5:
        polished2 = _tight_polish(best_params, start_time)
        px2 = polished2[:N]
        py2 = polished2[N:2*N]
        pr2 = polished2[2*N:3*N]
        
        pr2_lp = _optimize_radii_lp(px2, py2)
        px2, py2, pr2_lp = _make_valid(px2, py2, pr2_lp)
        p2_lp = _sum_r(pr2_lp)
        
        px2b, py2b, pr2b = _make_valid(px2, py2, pr2)
        p2_soft = _sum_r(pr2b)
        
        if p2_lp >= p2_soft:
            p2_score = p2_lp
            p2_params = np.concatenate([px2, py2, pr2_lp])
        else:
            p2_score = p2_soft
            p2_params = np.concatenate([px2b, py2b, pr2b])
        
        if p2_score > best_score:
            best_score = p2_score
            best_params = p2_params
    
    if best_params is None:
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
