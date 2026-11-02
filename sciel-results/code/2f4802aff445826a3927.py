import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 138.0

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

def _objective(params, start):
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
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-8))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start):
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
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-12))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-12))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 100.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_logsumexp(params, start, beta=20.0):
    """Log-sum-exp based smooth surrogate for the constraint violation."""
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
    violations = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    v_pairs = violations[mask]
    v_max = np.max(v_pairs)
    if v_max > 0:
        lse = v_max + np.log(np.sum(np.exp(beta * (v_pairs - v_max)))) / beta
    else:
        lse = 0.0

    bviol = np.maximum(0, r - x) + np.maximum(0, x - (1.0 - r)) + \
            np.maximum(0, r - y) + np.maximum(0, y - (1.0 - r))
    b_max = np.max(bviol)
    if b_max > 0:
        lse_b = b_max + np.log(np.sum(np.exp(beta * (bviol - b_max)))) / beta
    else:
        lse_b = 0.0

    mu = 200.0
    return obj + mu * (lse + lse_b)

def _optimize_radii_lp(x, y):
    """
    Solve for maximum radii given fixed centers using Linear Programming.
    Maximize sum(r) subject to r_i + r_j <= d_ij and r_i <= dist_to_bound.
    """
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(min(x[i], 1.0-x[i], y[i], 1.0-y[i]))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, 0.5)]*N, method='highs')
        if res.success:
            r = res.x
            # Small epsilon to ensure strict inequality for safety
            r = np.maximum(0, r * (1 - 1e-10))
            return r
    except Exception:
        pass
    
    # Fallback: greedy
    r = np.zeros(N)
    for i in range(N):
        r[i] = min(x[i], 1.0-x[i], y[i], 1.0-y[i])
        for j in range(N):
            if i != j:
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                r[i] = min(r[i], max(0, d - r[j]))
    return r

def _physics_optimize(x, y, r, start, max_iter=2000):
    """Physics-based optimization using repulsive forces."""
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)
    r = r.copy().astype(np.float64)
    
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    lr = 0.01
    force_scale = 50.0
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-10:
                    ang = (i * 0.7 + j * 1.3) % (2 * np.pi)
                    dx, dy = np.cos(ang), np.sin(ang)
                    d = 1e-10
                need = r[i] + r[j]
                if d < need:
                    overlap = need - d
                    f = force_scale * overlap
                    fx[i] += f * dx / d
                    fy[i] += f * dy / d
                    fx[j] -= f * dx / d
                    fy[j] -= f * dy / d
                    fr[i] += f * 0.1
                    fr[j] += f * 0.1
                else:
                    gap = d - need
                    f = force_scale * 0.01 * gap
                    fr[i] += f
                    fr[j] += f
        
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += force_scale * (r[i] - x[i])
            elif x[i] > 1.0 - r[i]:
                fx[i] -= force_scale * (x[i] - (1.0 - r[i]))
            if y[i] < r[i]:
                fy[i] += force_scale * (r[i] - y[i])
            elif y[i] > 1.0 - r[i]:
                fy[i] -= force_scale * (y[i] - (1.0 - r[i]))
            margin_x = min(x[i] - r[i], 1.0 - x[i] - r[i])
            margin_y = min(y[i] - r[i], 1.0 - y[i] - r[i])
            if margin_x > 0.01 and margin_y > 0.01:
                fr[i] += force_scale * 0.05 * min(margin_x, margin_y)
        
        x += lr * fx
        y += lr * fy
        r += lr * fr * 0.1
        
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        r = np.maximum(r, 1e-4)
        r = np.minimum(r, 0.45)
        
        if _is_valid(x, y, r):
            score = _sum_r(r)
            if score > best_score:
                best_score = score
                best = (x.copy(), y.copy(), r.copy())
        
        if it % 100 == 99:
            lr *= 0.95
    
    return best

def _push_apart(x, y, r, iters=500):
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = (dist < need) & (np.abs(np.triu(np.ones((N, N), dtype=bool), k=1).astype(float)) > 0)
        
        if not np.any(overlaps):
            break
        
        disp_x = np.zeros(N)
        disp_y = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d = np.hypot(dx_ij, dy_ij)
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx_ij / d, dy_ij / d
                    gap = need_ij - d
                    step = 0.5 * gap
                    disp_x[i] += step * ux
                    disp_y[i] += step * uy
                    disp_x[j] -= step * ux
                    disp_y[j] -= step * uy
        
        x += disp_x
        y += disp_y
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    return x, y

def _make_valid(x, y, r):
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(50):
        x, y = _push_apart(x, y, r, iters=200)
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
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
    
    return x, y, r

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
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

def _voronoi_init():
    rng = np.random.RandomState(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    try:
        dists = np.sqrt((x[:,None]-x[None,:])**2 + (y[:,None]-y[None,:])**2)
        np.fill_diagonal(dists, np.inf)
        r = 0.45 * np.min(dists, axis=1)
        r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    except Exception:
        r = np.full(N, 0.05)
    return x, y, r

def _lbfgsb_optimize(x, y, r, start, max_iter=3000):
    """L-BFGS-B with log-sum-exp smooth surrogate."""
    if _elapsed(start) > TIME_LIMIT - 10:
        return x, y, r
    
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    p0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.45))
    
    try:
        res = minimize(
            _objective_logsumexp, p0, args=(start,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': 10000}
        )
        vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
        if _is_valid(vx, vy, vr):
            s = _sum_r(vr)
            if s > best_score:
                best = (vx, vy, vr)
                best_score = s
    except Exception:
        pass
    
    return best

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # L-BFGS-B with log-sum-exp
    if _elapsed(start) < TIME_LIMIT - 15:
        try:
            vx, vy, vr = _lbfgsb_optimize(x, y, r, start, max_iter=2000)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Powell with standard objective
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Powell with tight objective
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-14}
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

def _tight_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # L-BFGS-B with very tight log-sum-exp
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.001, 0.45))
            res = minimize(
                _objective_logsumexp, p0, args=(start, 50.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-10, 'maxfun': 20000}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Powell with tight objective
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-15}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Nelder-Mead with tight objective
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Fine-grained local perturbation
    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(42)
        for trial in range(80):
            if _elapsed(start) > TIME_LIMIT - 1:
                break
            scale = 0.003 * (1.0 - trial / 80.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = best[2] + rng.normal(0, scale * 0.3, N)
            pr = np.maximum(pr, 1e-4)
            
            vx, vy, vr = _make_valid(px, py, pr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
    
    return best

def _grow_radii(x, y, r, start, max_rounds=50):
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        if _elapsed(start) > TIME_LIMIT - 2:
            break
        
        improved = False
        for i in range(N):
            slack_boundary = min(x[i] - r[i], 1.0 - x[i] - r[i],
                                  y[i] - r[i], 1.0 - y[i] - r[i])
            
            max_dr_pair = 1e9
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                slack = d - r[i] - r[j]
                if slack < max_dr_pair:
                    max_dr_pair = slack
            
            max_dr = min(slack_boundary, max_dr_pair)
            if max_dr > 1e-10:
                r[i] += max_dr * 0.995
                improved = True
        
        if not improved:
            break
        
        x, y = _push_apart(x, y, r, iters=150)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        if not _is_valid(x, y, r):
            r = np.minimum(r, 0.995 * r)
            x, y = _push_apart(x, y, r, iters=150)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
    
    return x, y, r

def _final_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Physics-based optimization
    if _elapsed(start) < TIME_LIMIT - 15:
        try:
            vx, vy, vr = _physics_optimize(*best, start, max_iter=3000)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # L-BFGS-B with high beta log-sum-exp
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.001, 0.45))
            res = minimize(
                _objective_logsumexp, p0, args=(start, 100.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-10, 'maxfun': 20000}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Tight polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = _tight_polish(*best, start)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best = res
                    best_score = s
        except Exception:
            pass
    
    # Grow radii
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            vx, vy, vr = _grow_radii(*best, start, max_rounds=40)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # Local perturbation
    if _elapsed(start) < TIME_LIMIT - 2:
        rng = np.random.RandomState(123)
        for _ in range(60):
            if _elapsed(start) > TIME_LIMIT - 2:
                break
            scale = 0.008 * (1 - _ / 60.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = best[2] + rng.normal(0, scale * 0.5, N)
            pr = np.maximum(pr, 1e-4)
            
            vx, vy, vr = _make_valid(px, py, pr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
    
    return best

def construct():
    start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_rect_init(9, 4))
    inits.append(_rect_init(4, 9))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_voronoi_init())
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])
    
    # Optimize each initialization with time budget
    n_inits = len(inits)
    
    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 25:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res
    
    # Final high-precision polish
    if _elapsed(start) < TIME_LIMIT - 10:
        res = _final_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res
                best_score = s
    
    # One more round of radius growth
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            vx, vy, vr = _grow_radii(*best, start, max_rounds=30)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # LP-based radius optimization for the best center configuration
    if _elapsed(start) < TIME_LIMIT - 1:
        try:
            x_best, y_best, _ = best
            r_lp = _optimize_radii_lp(x_best, y_best)
            # Ensure validity after LP
            x_valid, y_valid, r_valid = _make_valid(x_best, y_best, r_lp)
            if _is_valid(x_valid, y_valid, r_valid):
                s = _sum_r(r_valid)
                if s > best_score:
                    best = (x_valid, y_valid, r_valid)
                    best_score = s
        except Exception:
            pass
    
    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
