import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 130.0  # safety margin before 150s kill

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
    r = np.maximum(params[2*N:3*N], 0.0)
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

    mu = 80.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start):
    """Higher penalty for final polish phase."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = np.maximum(params[2*N:3*N], 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-10))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-10))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 200.0
    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iters=300):
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need
        if not np.any(overlaps):
            break
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                d = np.hypot(dx[i,j], dy[i,j])
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx[i,j]/d, dy[i,j]/d
                    step = 0.5 * (need_ij - d)
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

def _max_radii_for_centers(x, y, iters=200):
    """Given fixed centers, compute the maximum feasible radii via iterative relaxation."""
    # Boundary limits
    bound_r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    r = bound_r.copy()
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    
    for _ in range(iters):
        new_r = bound_r.copy()
        for i in range(N):
            for j in range(N):
                if i != j:
                    limit = dist[i, j] - r[j]
                    if limit < new_r[i]:
                        new_r[i] = limit
            new_r[i] = max(0.0, min(new_r[i], bound_r[i]))
        if np.allclose(new_r, r, rtol=1e-9, atol=1e-10):
            r = new_r
            break
        r = new_r
    return r

def _make_valid(x, y, r):
    """Guarantee validity and maximize radii for given centers."""
    # 1. Resolve overlaps and bounds
    for _ in range(5):
        x, y = _push_apart(x, y, r, iters=100)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    
    # 2. Shrink to validity if needed
    for _ in range(100):
        ok = True
        for i in range(N):
            mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > mr:
                r[i] = max(0.0, mr * 0.999)
                ok = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need - 1e-9
        if np.any(overlaps):
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
            ok = False
        if ok:
            break
    
    # 3. Expand to maximal feasible radii
    r = _max_radii_for_centers(x, y, iters=300)
    
    # 4. Final safety
    for _ in range(50):
        ok = True
        for i in range(N):
            mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > mr + 1e-9:
                r[i] = mr
                ok = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need - 1e-9
        if np.any(overlaps):
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.995
            r[j] *= 0.995
            ok = False
        if ok:
            break
    
    return x, y, r

def _phyllotaxis_init():
    """Phyllotaxis (sunflower) pattern - low discrepancy, good for circle packing."""
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        r = 0.45 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x[i] = 0.5 + r * np.cos(theta)
        y[i] = 0.5 + r * np.sin(theta)
    # Compute radii based on nearest neighbor distance
    r = np.zeros(n)
    for i in range(n):
        mn = 1e9
        for j in range(n):
            if i != j:
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                mn = min(mn, d)
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
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
    r = np.full(N, 0.06)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _coordinate_descent(x, y, r, start, max_iter=5000):
    """Coordinate descent: alternate between optimizing positions and radii."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
        
        # Update radii given current centers
        r = _max_radii_for_centers(x, y, iters=50)
        
        # Update positions: for each circle, try to move it to increase min radius
        # Use a simple gradient-free approach: try small perturbations
        improved = False
        for i in range(N):
            best_x, best_y = x[i], y[i]
            best_r_i = r[i]
            
            # Try a few random perturbations
            for _ in range(20):
                if _elapsed(start) > TIME_LIMIT - 5:
                    break
                dx = np.random.uniform(-0.03, 0.03)
                dy = np.random.uniform(-0.03, 0.03)
                nx = np.clip(x[i] + dx, r[i], 1.0 - r[i])
                ny = np.clip(y[i] + dy, r[i], 1.0 - r[i])
                
                # Check if this improves the radius of circle i
                # by checking constraints with all other circles
                min_limit = min(nx, 1.0 - nx, ny, 1.0 - ny)
                for j in range(N):
                    if i != j:
                        d = np.hypot(nx - x[j], ny - y[j])
                        limit = d - r[j]
                        if limit < min_limit:
                            min_limit = limit
                if min_limit > best_r_i + 1e-8:
                    best_x, best_y = nx, ny
                    best_r_i = min_limit
                    improved = True
            
            x[i] = best_x
            y[i] = best_y
        
        if not improved and it > 100:
            # Try a larger random kick to escape local minima
            if it % 200 == 0:
                idx = np.random.randint(N)
                x[idx] = np.random.uniform(0.1, 0.9)
                y[idx] = np.random.uniform(0.1, 0.9)
                r = _max_radii_for_centers(x, y, iters=50)
    
    return x, y, r

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = minimize(
                _objective, params, args=(start,),
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
    
    # Coordinate descent refinement
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            vx, vy, vr = _coordinate_descent(best[0], best[1], best[2], start, max_iter=3000)
            vx, vy, vr = _make_valid(vx, vy, vr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Tight Powell polish
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12}
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
    """Final aggressive polish: coordinate descent with very small steps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(2000):
        if _elapsed(start) > TIME_LIMIT - 2:
            break
        
        r = _max_radii_for_centers(x, y, iters=100)
        
        improved = False
        for i in range(N):
            if _elapsed(start) > TIME_LIMIT - 2:
                break
            best_x, best_y = x[i], y[i]
            best_r_i = r[i]
            
            # Fine-grained search
            for dx in np.linspace(-0.01, 0.01, 5):
                for dy in np.linspace(-0.01, 0.01, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx = np.clip(x[i] + dx, r[i], 1.0 - r[i])
                    ny = np.clip(y[i] + dy, r[i], 1.0 - r[i])
                    
                    min_limit = min(nx, 1.0 - nx, ny, 1.0 - ny)
                    for j in range(N):
                        if i != j:
                            d = np.hypot(nx - x[j], ny - y[j])
                            limit = d - r[j]
                            if limit < min_limit:
                                min_limit = limit
                    if min_limit > best_r_i + 1e-10:
                        best_x, best_y = nx, ny
                        best_r_i = min_limit
                        improved = True
            
            x[i] = best_x
            y[i] = best_y
        
        r = _max_radii_for_centers(x, y, iters=100)
        
        if not improved and it > 200:
            break
    
    x, y, r = _make_valid(x, y, r)
    return x, y, r

def construct():
    start = time.time()
    inits = []
    inits.append(_phyllotaxis_init())
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
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final tight polish on best
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
