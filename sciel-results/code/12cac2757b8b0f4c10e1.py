import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32
TIME_LIMIT = 135.0  # evaluator kills at 150s; keep 15s margin

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
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                grad[i] += -mu * dsmooth[i, j] * ux
                grad[N+i] += -mu * dsmooth[i, j] * uy
                grad[2*N+i] += mu * dsmooth[i, j]
                
    for i in range(N):
        grad[i] += mu * (-sm_dn_x[i] + sm_dp_x[i])
        grad[N+i] += mu * (-sm_dn_y[i] + sm_dp_y[i])
        grad[2*N+i] += -1.0
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

def _pbd_expand(x, y, r, start, max_iters=2000, growth=0.0005):
    """Position-based dynamics: grow radii and project positions to maintain feasibility."""
    if _elapsed(start) > TIME_LIMIT - 10:
        return x, y, r
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for it in range(max_iters):
        if _elapsed(start) > TIME_LIMIT - 8:
            break
        # Grow all radii slightly
        r += growth
        # Project positions to resolve overlaps and boundary violations
        for _ in range(30):
            moved = False
            # Boundary projection
            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]; moved = True
                elif x[i] > 1.0 - r[i]:
                    x[i] = 1.0 - r[i]; moved = True
                if y[i] < r[i]:
                    y[i] = r[i]; moved = True
                elif y[i] > 1.0 - r[i]:
                    y[i] = 1.0 - r[i]; moved = True
            # Pairwise overlap resolution
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    need = r[i] + r[j]
                    if d < need:
                        if d < 1e-12:
                            ang = (i * 0.7 + j * 1.3) % (2 * np.pi)
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
            if not moved:
                break
        # If we can't resolve, shrink back
        if not _is_valid(x, y, r):
            r -= growth * 2
            x, y, r = _make_valid(x, y, r)
            if _is_valid(x, y, r):
                break
    return x, y, r

def _delaunay_refine(x, y, r, start, n_iters=50):
    """Use Delaunay triangulation to identify contact graph, then refine."""
    if _elapsed(start) > TIME_LIMIT - 5:
        return x, y, r
    
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Build Delaunay triangulation
    try:
        points = np.column_stack([x, y])
        tri = Delaunay(points)
    except Exception:
        return x, y, r
    
    # Get edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = simplex[i], simplex[j]
                if a > b: a, b = b, a
                edges.add((a, b))
    
    # Add edges for circles that are close (potential contacts)
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            if d < (r[i] + r[j]) * 1.5:
                edges.add((i, j))
    
    # Iteratively adjust radii and positions based on contact constraints
    for it in range(n_iters):
        if _elapsed(start) > TIME_LIMIT - 3:
            break
        
        # For each edge, check if circles are in contact or overlapping
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            need = r[i] + r[j]
            
            if d < 1e-12:
                continue
            
            if d < need:
                # Overlapping: push apart and/or reduce radii
                ux, uy = dx/d, dy/d
                overlap = need - d
                # Reduce radii proportionally
                r[i] *= (1 - overlap * 0.1)
                r[j] *= (1 - overlap * 0.1)
                # Push apart
                step = 0.3 * overlap
                x[i] += step * ux
                y[i] += step * uy
                x[j] -= step * ux
                y[j] -= step * uy
            elif d > need * 1.1:
                # Not in contact: try to increase radii if space allows
                # Check if increasing radii would cause overlap with neighbors
                can_increase = True
                for k in range(N):
                    if k == i or k == j:
                        continue
                    dk_i = np.hypot(x[i]-x[k], y[i]-y[k])
                    dk_j = np.hypot(x[j]-x[k], y[j]-y[k])
                    if dk_i < r[i] + r[k] + 0.01 or dk_j < r[j] + r[k] + 0.01:
                        can_increase = False
                        break
                if can_increase:
                    # Increase radii slightly
                    increase = min(0.001, (d - need) * 0.1)
                    r[i] += increase
                    r[j] += increase
        
        # Clamp to bounds
        for i in range(N):
            mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > mr * 0.999:
                r[i] = max(0.0, mr * 0.998)
        
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    
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
            
    # 3. PBD expansion
    if _elapsed(start) < TIME_LIMIT - 8:
        try:
            vx, vy, vr = _pbd_expand(best[0], best[1], best[2], start, max_iters=1500, growth=0.0004)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # 4. Delaunay-based refinement
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            vx, vy, vr = _delaunay_refine(best[0], best[1], best[2], start, n_iters=30)
            vx, vy, vr = _make_valid(vx, vy, vr)
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
    
    # PBD expansion in final polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            vx, vy, vr = _pbd_expand(best[0], best[1], best[2], start, max_iters=1000, growth=0.0003)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # Local perturbation
    if _elapsed(start) < TIME_LIMIT - 2:
        rng = np.random.RandomState(123)
        for _ in range(20):
            if _elapsed(start) > TIME_LIMIT - 2: break
            scale = 0.01 * (1 - _ / 20.0)
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

    # Final Delaunay refinement
    if _elapsed(start) < TIME_LIMIT - 0.5:
        try:
            vx, vy, vr = _delaunay_refine(best[0], best[1], best[2], start, n_iters=30)
            vx, vy, vr = _make_valid(vx, vy, vr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

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
