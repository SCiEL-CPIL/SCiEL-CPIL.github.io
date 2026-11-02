import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32
TIME_LIMIT = 135.0  # evaluator kills at 150s; keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    # bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # pairwise disjoint interiors
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

    # Vectorized overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    # Boundary penalties
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-8))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iters=500):
    """Move centers to resolve overlaps; never shrink radii. Keep in bounds."""
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        # Vectorized check for overlaps
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
                    # Move each half the gap along the separating axis
                    step = 0.5 * gap
                    x[i] += step * ux
                    y[i] += step * uy
                    x[j] -= step * ux
                    y[j] -= step * uy
                    moved = True
        # Clamp back into square (respecting radius)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if not moved:
            break
    return x, y

def _physics_optimize(x, y, r, start, max_iter=2000):
    """
    Position-Based Dynamics (PBD) for circle packing.
    Iteratively: (1) grow radii, (2) project to satisfy non-overlap and
    boundary constraints. This turns optimization into a physics-stabilization
    process.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    best_score = _sum_r(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    # Growth rate: how much to try to expand radii each iteration
    growth = 0.002
    # Number of projection passes per iteration for convergence
    proj_passes = 30
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 5:
            break
        
        # Step 1: Grow all radii slightly
        r = r + growth
        
        # Cap radii by boundary distance (can't exceed half the min distance to wall)
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > max_r:
                r[i] = max_r
        
        # Step 2: Project to satisfy constraints (position-based dynamics)
        for _ in range(proj_passes):
            # Project pairwise non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    need = r[i] + r[j]
                    if d < need:
                        if d < 1e-12:
                            # Degenerate: push in arbitrary direction
                            ang = (i * 2.399 + j * 1.618) % (2 * np.pi)
                            ux, uy = np.cos(ang), np.sin(ang)
                        else:
                            ux, uy = dx / d, dy / d
                        # Distribute correction proportional to radii (heavier moves less)
                        wi = r[j] / (r[i] + r[j] + 1e-12)
                        wj = r[i] / (r[i] + r[j] + 1e-12)
                        gap = need - d
                        x[i] += wi * gap * ux
                        y[i] += wi * gap * uy
                        x[j] -= wj * gap * ux
                        y[j] -= wj * gap * uy
            
            # Project boundary constraints
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    x[i] = r[i]
                # Right wall
                if x[i] > 1.0 - r[i]:
                    x[i] = 1.0 - r[i]
                # Bottom wall
                if y[i] < r[i]:
                    y[i] = r[i]
                # Top wall
                if y[i] > 1.0 - r[i]:
                    y[i] = 1.0 - r[i]
        
        # Check validity and score
        if _is_valid(x, y, r):
            score = _sum_r(r)
            if score > best_score + 1e-10:
                best_score = score
                best_state = (x.copy(), y.copy(), r.copy())
        
        # Adaptive growth: slow down if we're not making progress
        # (implicit in the projection - if radii can't grow, they stay)
    
    bx, by, br = best_state
    bx, by, br = _make_valid(bx, by, br)
    return bx, by, br

def _make_valid(x, y, r):
    """Guarantee validity: push apart, then shrink any residual overlap/bounds."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Iteratively push apart and clamp
    for _ in range(50):
        x, y = _push_apart(x, y, r, iters=200)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        # Check if valid
        valid = _is_valid(x, y, r)
        if valid:
            break
    
    if not _is_valid(x, y, r):
        # Aggressive shrinking to ensure validity
        for _ in range(500):
            # Fix bounds first
            for i in range(N):
                mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > mr * 0.999:
                    r[i] = max(0.0, mr * 0.998)
            
            # Fix overlaps
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need - 1e-10
            
            if not np.any(overlaps):
                break
            
            # Shrink the most overlapping pair
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
    
    return x, y, r

def _hex_init():
    # Hexagonal packing for 32 circles, then scale to fit square
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
    # radius = half min neighbor distance, capped by boundary
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

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Physics-based optimization first
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            vx, vy, vr = _physics_optimize(x, y, r, start, max_iter=1000)
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
    
    # Powell polish on best
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 1500, 'ftol': 1e-10}
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

def _objective_tight(params, start):
    """Tighter penalty objective for final polishing with smaller smoothing constants."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    # Vectorized overlap penalty with tighter smoothing
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-12))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    # Boundary penalties with tighter smoothing
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-12))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 100.0
    return obj + mu * (overlap_pen + bound_pen)

def _tight_polish(x, y, r, start):
    """High-precision polish using the tighter objective with very tight tolerances."""
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Powell with tight objective and very tight tolerance
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-14, 'xtol': 1e-14}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Second pass with even tighter settings
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Fine-grained local perturbation with tight validation
    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(42)
        for trial in range(100):
            if _elapsed(start) > TIME_LIMIT - 1:
                break
            scale = 0.005 * (1.0 - trial / 100.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = best[2] + rng.normal(0, scale * 0.3, N)
            pr = np.maximum(pr, 1e-4)
            
            # Quick validity check before expensive optimization
            if _is_valid(px, py, pr):
                vx, vy, vr = _make_valid(px, py, pr)
                if _is_valid(vx, vy, vr):
                    s = _sum_r(vr)
                    if s > best_score:
                        best_score = s
                        best = (vx, vy, vr)
    
    return best

def _final_polish(x, y, r, start):
    """High-precision final polish using multiple optimization methods."""
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # First: standard Powell passes
    for ftol in [1e-10, 1e-12]:
        if _elapsed(start) > TIME_LIMIT - 8:
            break
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 2000, 'ftol': ftol}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Then: dedicated tight polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = _tight_polish(*best, start)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass
    
    # Local perturbation search
    if _elapsed(start) < TIME_LIMIT - 2:
        rng = np.random.RandomState(123)
        for _ in range(50):
            if _elapsed(start) > TIME_LIMIT - 2:
                break
            scale = 0.01 * (1 - _ / 50.0)
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

def _voronoi_init():
    """
    Generate a packing based on Voronoi cells of a random point set,
    then optimize the centers/radii.
    """
    rng = np.random.RandomState(42)
    # Generate points in a grid-like random pattern
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    
    # Compute Voronoi diagram to find inradii
    try:
        tri = Delaunay(np.column_stack((x, y)))
        # For each point, find its Voronoi cell inradius
        # This is complex, so we approximate by distance to nearest neighbors
        r = np.zeros(N)
        for i in range(N):
            # Distance to nearest neighbors
            dists = np.sqrt((x[i]-x)**2 + (y[i]-y)**2)
            dists[i] = np.inf
            # Take the 3 nearest neighbors to get a robust radius estimate
            nearest = np.argsort(dists)[:3]
            min_dist = np.min(dists[nearest])
            r[i] = 0.45 * min_dist
            # Cap by boundary
            r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    except Exception:
        r = np.full(N, 0.05)
        
    return x, y, r

def construct():
    start = time.time()
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

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 15:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final high-precision polish
    if _elapsed(start) < TIME_LIMIT - 5:
        res = _final_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    x, y, r = best
    # ensure final validity
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
