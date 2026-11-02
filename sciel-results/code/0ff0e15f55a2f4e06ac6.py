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
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist[mask] < min_dist[mask] - eps):
        return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _optimize_radii_lp(x, y):
    """Solve for maximum radii given fixed centers using Linear Programming."""
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
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
        if res.success and res.x is not None:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
        
    # Fallback: iterative shrink
    r = np.array([min(x[i], 1.0-x[i], y[i], 1.0-y[i]) for i in range(N)])
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if r[i] + r[j] > d:
                    shrink = (d / (r[i] + r[j])) * 0.999 if (r[i] + r[j]) > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed: 
            break
    return r

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

def _objective(params, start, mu=500.0, soft_eps=1e-6):
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

def _objective_tight(params, start, mu=2000.0, soft_eps=1e-8):
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

def _push_apart(x, y, r, iters=200):
    """Move centers to resolve overlaps; never shrink radii. Keep in bounds."""
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                need = r[i] + r[j]
                if d < need:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / d, dy / d
                    gap = need - d
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

def _hex_init():
    rows = [6, 5, 6, 5, 6, 4]
    x = []
    y = []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            x.append(cx)
            y.append(ry)
    x = np.array(x[:N])
    y = np.array(y[:N])
    while len(x) < N:
        x = np.append(x, 0.5)
        y = np.append(y, 0.5)
    r = _optimize_radii_lp(x, y)
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
    r = _optimize_radii_lp(x, y)
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = _optimize_radii_lp(x, y)
    return x, y, r

def _pbd_simulate(x, y, r, start, n_steps=5000, expand_rate=0.0001):
    """Position-Based Dynamics simulation to expand circles and resolve overlaps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # First, make sure initial state is valid
    x, y, r = _make_valid(x, y, r, max_iter=50)
    
    for step in range(n_steps):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
        
        # Try to expand all radii slightly
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            r[i] = min(r[i] + expand_rate, max_r * 0.9999)
        
        # Resolve overlaps via position-based dynamics
        for _ in range(10):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    need = r[i] + r[j]
                    if d < need - 1e-10:
                        if d < 1e-12:
                            ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                            ux, uy = np.cos(ang), np.sin(ang)
                        else:
                            ux, uy = dx / d, dy / d
                        gap = need - d
                        # Move both circles apart
                        x[i] += 0.5 * gap * ux
                        y[i] += 0.5 * gap * uy
                        x[j] -= 0.5 * gap * ux
                        y[j] -= 0.5 * gap * uy
                        moved = True
            
            # Keep in bounds
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - y[i])
            
            if not moved:
                break
        
        # If we can't expand any more, try smaller expansion rate
        if step > 0 and step % 500 == 0:
            expand_rate *= 0.5
    
    return x, y, r

def _optimize_centers(x, y, r, start):
    """Optimize centers and radii using PBd simulation, then Powell polish."""
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # PBd simulation
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            vx, vy, vr = _pbd_simulate(x, y, r, start)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Powell polish
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 1500, 'ftol': 1e-12}
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

def _lp_refine(x, y, r, start, n_iters=3):
    """Iteratively refine: solve LP for optimal radii, then PBd + Powell to move centers."""
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = _sum_r(r)
    
    for it in range(n_iters):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
        
        # Solve LP for optimal radii given current centers
        r_lp = _optimize_radii_lp(best_x, best_y)
        if _is_valid(best_x, best_y, r_lp) and _sum_r(r_lp) > best_score:
            best_r = r_lp.copy()
            best_score = _sum_r(r_lp)
        
        # Use PBd to try to expand further by moving centers
        if _elapsed(start) > TIME_LIMIT - 4:
            break
        try:
            vx, vy, vr = _pbd_simulate(best_x, best_y, best_r, start, n_steps=2000, expand_rate=0.00005)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best_x, best_y, best_r = vx.copy(), vy.copy(), vr.copy()
        except Exception:
            pass
        
        # Powell polish
        if _elapsed(start) > TIME_LIMIT - 3:
            break
        try:
            params = np.concatenate([best_x, best_y, best_r])
            res = minimize(
                _objective, params, args=(start,),
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best_x, best_y, best_r = vx.copy(), vy.copy(), vr.copy()
        except Exception:
            pass
    
    return best_x, best_y, best_r

def _tight_polish(x, y, r, start):
    """High-precision final polish."""
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Tight Powell
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            params = np.concatenate([x, y, r])
            res = minimize(
                _objective_tight, params, args=(start,),
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
    
    # Final LP solve
    if _elapsed(start) < TIME_LIMIT - 1:
        r_lp = _optimize_radii_lp(best[0], best[1])
        if _is_valid(best[0], best[1], r_lp):
            s = _sum_r(r_lp)
            if s > best_score:
                best = (best[0], best[1], r_lp)
    
    return best

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_random_init(42))

    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 15:
            break
        # Optimize centers+radii with PBd + Powell
        res = _optimize_centers(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # PBd + LP refinement on best
    if _elapsed(start) < TIME_LIMIT - 8:
        res = _lp_refine(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res
                best_score = s

    # Final tight polish
    if _elapsed(start) < TIME_LIMIT - 3:
        res = _tight_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
