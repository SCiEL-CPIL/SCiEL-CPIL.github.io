import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
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
            # Bound constraints
            r[i] = min(r[i], x[i] * 0.999, (1.0 - x[i]) * 0.999,
                       y[i] * 0.999, (1.0 - y[i]) * 0.999)
            if r[i] < 1e-6:
                r[i] = 1e-6
        # Pairwise overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    # Shrink both proportionally
                    shrink = (d / min_d) * 0.999 if min_d > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _hex_init():
    """Hexagonal packing initialization for 32 circles."""
    rows = [6, 5, 6, 5, 6, 4]  # 32 total
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

def _compute_radii_from_centers(x, y):
    """
    Given circle centers, compute maximum possible radii such that circles don't overlap
    and stay within bounds. Uses LP-like approach: for each pair, r_i + r_j <= dist(i,j),
    and for boundaries: r_i <= min(x_i, 1-x_i, y_i, 1-y_i).
    We solve this iteratively using a simple relaxation method.
    """
    r = np.zeros(N)
    # Initial upper bounds from boundaries
    for i in range(N):
        r[i] = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
    
    # Iterative relaxation
    for _ in range(200):
        changed = False
        for i in range(N):
            # Boundary constraint
            max_r_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > max_r_bound:
                r[i] = max_r_bound
                changed = True
            # Pairwise constraints
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                # r[i] + r[j] <= d
                if r[i] + r[j] > d:
                    # Reduce proportionally
                    excess = r[i] + r[j] - d
                    if excess > 1e-12:
                        # Split reduction based on current radii
                        total_r = r[i] + r[j]
                        if total_r < 1e-12:
                            r[i] = d / 2
                            r[j] = d / 2
                        else:
                            r[i] -= excess * (r[i] / total_r)
                            r[j] -= excess * (r[j] / total_r)
                        r[i] = max(r[i], 0)
                        r[j] = max(r[j], 0)
                        changed = True
        if not changed:
            break
    
    # Ensure non-negative and within bounds
    r = np.maximum(r, 0)
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
    
    return r

def _voronoi_objective(params):
    """
    Objective for Voronoi-based optimization.
    params: [x_0, ..., x_{N-1}, y_0, ..., y_{N-1}]
    Radii are computed from centers using the relaxation method.
    We maximize sum of radii (minimize negative sum).
    """
    x = params[:N]
    y = params[N:2*N]
    
    r = _compute_radii_from_centers(x, y)
    
    # Check validity
    valid = True
    for i in range(N):
        if x[i] < -1e-7 or x[i] > 1.0 + 1e-7:
            valid = False
            break
        if y[i] < -1e-7 or y[i] > 1.0 + 1e-7:
            valid = False
            break
        if r[i] < -1e-7:
            valid = False
            break
        if x[i] < r[i] - 1e-7 or x[i] > 1.0 - r[i] + 1e-7:
            valid = False
            break
        if y[i] < r[i] - 1e-7 or y[i] > 1.0 - r[i] + 1e-7:
            valid = False
            break
    if not valid:
        return 1e6
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < (r[i] + r[j]) - 1e-7:
                return 1e6
    
    return -np.sum(r)

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

def _objective(params, soft_eps=1e-4):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    obj = -np.sum(r)
    
    # Pairwise overlap penalty (soft max)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    # Soft max of positive part
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    # Boundary penalties (soft max)
    pen = np.zeros(4*N)
    pen[0:N] = r - x          # left
    pen[N:2*N] = x - (1.0 - r)  # right
    pen[2*N:3*N] = r - y       # bottom
    pen[3*N:4*N] = y - (1.0 - r)  # top
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

def _push_apart(x, y, r, iterations=80):
    """Physics-based push apart with radius adaptation to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for it in range(iterations):
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
                    # Push apart
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        # Clamp to bounds and shrink radii if needed
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i]) * 0.999
            if r[i] > max_r:
                r[i] = max(max_r, 1e-6)
            x[i] = np.clip(x[i], r[i]*0.999, 1.0 - r[i]*0.999)
            y[i] = np.clip(y[i], r[i]*0.999, 1.0 - r[i]*0.999)
        if not moved:
            break
    return x, y, r

def _optimize(x0, start_time, time_budget=100):
    """Run L-BFGS-B optimization."""
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 50000})
        return res.x
    except Exception:
        return x0

def _optimize_voronoi(x0, start_time, time_budget=100):
    """
    Optimize using Voronoi-based formulation.
    x0: array of shape (2*N,) with [x_coords, y_coords]
    Returns: (x, y, r) arrays
    """
    def _voronoi_obj(params):
        x = params[:N]
        y = params[N:2*N]
        r = _compute_radii_from_centers(x, y)
        
        # Penalty for invalid positions
        pen = 0.0
        for i in range(N):
            if x[i] < 0 or x[i] > 1 or y[i] < 0 or y[i] > 1:
                pen += 10.0 * (abs(min(0, x[i])) + abs(max(0, x[i] - 1)) + 
                              abs(min(0, y[i])) + abs(max(0, y[i] - 1)))
        
        # Check pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < (r[i] + r[j]) - 1e-10:
                    pen += 10.0 * ((r[i] + r[j]) - d)
        
        return -np.sum(r) + pen
    
    try:
        res = minimize(_voronoi_obj, x0, method='L-BFGS-B',
                       bounds=[(0, 1)] * (2 * N),
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 100000})
        x = res.x[:N]
        y = res.x[N:2*N]
        r = _compute_radii_from_centers(x, y)
        return x, y, r
    except Exception:
        x = x0[:N]
        y = x0[N:2*N]
        r = _compute_radii_from_centers(x, y)
        return x, y, r

def _tight_polish(params, start_time, time_budget=10):
    """Final high-precision polish with tighter tolerances."""
    try:
        res = minimize(_objective_tight, params, method='L-BFGS-B',
                       options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15,
                                'maxfun': 200000})
        return res.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal
    hx, hy, hr = _hex_init()
    inits.append(np.concatenate([hx, hy, hr]))
    
    # Rectangular
    rx, ry, rr = _rect_init()
    inits.append(np.concatenate([rx, ry, rr]))
    
    # Multiple random inits with varied radius scales
    for seed, r_scale in [(42, 0.06), (123, 0.07), (456, 0.05), (789, 0.08), (101, 0.065), (202, 0.055), (303, 0.075)]:
        rng = np.random.RandomState(seed)
        rx2 = rng.uniform(0.1, 0.9, N)
        ry2 = rng.uniform(0.1, 0.9, N)
        rr2 = np.full(N, r_scale)
        inits.append(np.concatenate([rx2, ry2, rr2]))
    
    # Perturbed hex with varying perturbation magnitudes
    rng = np.random.RandomState(7)
    for pert in [0.03, 0.05, 0.08, 0.12]:
        hx2, hy2, hr2 = _hex_init()
        hx2 += rng.uniform(-pert, pert, N)
        hy2 += rng.uniform(-pert, pert, N)
        hx2 = np.clip(hx2, 0, 1)
        hy2 = np.clip(hy2, 0, 1)
        inits.append(np.concatenate([hx2, hy2, hr2]))
    
    # Perturbed rect
    rng2 = np.random.RandomState(13)
    rx3, ry3, rr3 = _rect_init()
    rx3 += rng2.uniform(-0.04, 0.04, N)
    ry3 += rng2.uniform(-0.04, 0.04, N)
    rx3 = np.clip(rx3, 0, 1)
    ry3 = np.clip(ry3, 0, 1)
    inits.append(np.concatenate([rx3, ry3, rr3]))
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Standard penalty-based optimization
    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 30:
            break
        
        x0a = x0[:N].copy()
        x0b = x0[N:2*N].copy()
        x0c = x0[2*N:3*N].copy()
        x0a, x0b, x0c = _push_apart(x0a, x0b, x0c, iterations=50)
        x0_opt = np.concatenate([x0a, x0b, x0c])
        
        result = _optimize(x0_opt, start_time)
        
        vx = result[:N]
        vy = result[N:2*N]
        vr = result[2*N:3*N]
        vx, vy, vr = _make_valid(vx, vy, vr)
        
        score = _sum_r(vr)
        
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx, vy, vr])
    
    # Phase 2: Voronoi-based optimization (structural change)
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 25:
        # Use best center positions as starting point for Voronoi optimization
        vx = best_params[:N].copy()
        vy = best_params[N:2*N].copy()
        voronoi_x0 = np.concatenate([vx, vy])
        
        # Run Voronoi-based optimization
        vrx, vry, vrr = _optimize_voronoi(voronoi_x0, start_time)
        vrx, vry, vrr = _make_valid(vrx, vry, vrr)
        voronoi_score = _sum_r(vrr)
        
        if voronoi_score > best_score:
            best_score = voronoi_score
            best_params = np.concatenate([vrx, vry, vrr])
    
    # Phase 3: Try Voronoi from different initializations
    if _elapsed(start_time) < TIME_LIMIT - 20:
        # Hex-based Voronoi
        hx, hy, _ = _hex_init()
        voronoi_x0_hex = np.concatenate([hx, hy])
        vhx, vhy, vhr = _optimize_voronoi(voronoi_x0_hex, start_time)
        vhx, vhy, vhr = _make_valid(vhx, vhy, vhr)
        hex_score = _sum_r(vhr)
        if hex_score > best_score:
            best_score = hex_score
            best_params = np.concatenate([vhx, vhy, vhr])
        
        # Rect-based Voronoi
        if _elapsed(start_time) < TIME_LIMIT - 15:
            rx, ry, _ = _rect_init()
            voronoi_x0_rect = np.concatenate([rx, ry])
            vrx2, vry2, vrr2 = _optimize_voronoi(voronoi_x0_rect, start_time)
            vrx2, vry2, vrr2 = _make_valid(vrx2, vry2, vrr2)
            rect_score = _sum_r(vrr2)
            if rect_score > best_score:
                best_score = rect_score
                best_params = np.concatenate([vrx2, vry2, vrr2])
    
    # Final tight polish on best
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        polished = _tight_polish(best_params, start_time)
        px = polished[:N]
        py = polished[N:2*N]
        pr = polished[2*N:3*N]
        px, py, pr = _make_valid(px, py, pr)
        p_score = _sum_r(pr)
        if p_score > best_score:
            best_score = p_score
            best_params = np.concatenate([px, py, pr])
    
    # Second polish round if time permits
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 10:
        polished2 = _tight_polish(best_params, start_time)
        px2 = polished2[:N]
        py2 = polished2[N:2*N]
        pr2 = polished2[2*N:3*N]
        px2, py2, pr2 = _make_valid(px2, py2, pr2)
        s2 = _sum_r(pr2)
        if s2 > best_score:
            best_score = s2
            best_params = np.concatenate([px2, py2, pr2])
    
    # Final local search refinement if time permits
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 5:
        fx = best_params[:N].copy()
        fy = best_params[N:2*N].copy()
        fr = best_params[2*N:3*N].copy()
        fx, fy, fr = _push_apart(fx, fy, fr, iterations=100)
        fx, fy, fr = _make_valid(fx, fy, fr)
        fs = _sum_r(fr)
        if fs > best_score:
            best_score = fs
            best_params = np.concatenate([fx, fy, fr])
    
    if best_params is None:
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
