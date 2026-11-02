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
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Boundary constraints
    for i in range(N):
        b = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(b)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * N, method='highs')
        if res.success:
            return np.clip(res.x, 1e-8, None)
    except Exception:
        pass
    
    # Fallback: greedy radius assignment
    r = np.zeros(N)
    for i in range(N):
        r[i] = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if r[i] + r[j] > d:
                    avg = d / 2.0
                    r[i] = min(r[i], avg)
                    r[j] = min(r[j], avg)
                    changed = True
        if not changed:
            break
    return np.clip(r, 1e-8, None)

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

def _center_only_optimize(x0, start_time):
    """
    Structural change: optimize only centers, then solve radii via LP.
    This decouples the non-linear center search from the radius assignment,
    allowing LP to find the exact optimal radii for any given center configuration.
    """
    x = x0[:N].copy()
    y = x0[N:2*N].copy()
    
    def center_obj(params):
        cx = params[:N]
        cy = params[N:2*N]
        # Objective: negative of LP-optimal sum of radii
        r = _optimize_radii_lp(cx, cy)
        return -np.sum(r)
    
    def center_grad(params):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(2*N)
        base = center_obj(params)
        for k in range(2*N):
            pert = params.copy()
            pert[k] += eps
            grad[k] = (center_obj(pert) - base) / eps
        return grad
    
    try:
        res = minimize(center_obj, x0[:2*N], method='L-BFGS-B',
                       jac=center_grad,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxfun': 5000})
        return res.x
    except Exception:
        return x0[:2*N]

def _full_lp_pipeline(x, y, start_time):
    """
    Full pipeline: given centers, solve LP for radii, then optionally
    do a joint refinement.
    """
    r = _optimize_radii_lp(x, y)
    return x, y, r

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
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
    
    # Phase 1: Center-only optimization with LP radius solving
    # This is the structural change: separate center search from radius assignment
    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 40:
            break
        
        x0a = x0[:N].copy()
        x0b = x0[N:2*N].copy()
        x0c = x0[2*N:3*N].copy()
        
        # Push apart centers to get a reasonable starting configuration
        x0a, x0b = _push_apart(x0a, x0b, x0c, iterations=30)
        
        # Center-only optimization
        centers = np.concatenate([x0a, x0b])
        opt_centers = _center_only_optimize(centers, start_time)
        
        ox = opt_centers[:N]
        oy = opt_centers[N:2*N]
        
        # Solve LP for optimal radii
        or_ = _optimize_radii_lp(ox, oy)
        
        # Make valid
        ox, oy, or_ = _make_valid(ox, oy, or_)
        
        score = _sum_r(or_)
        
        if score > best_score:
            best_score = score
            best_params = np.concatenate([ox, oy, or_])
    
    # Phase 2: Joint optimization from best center-LP result
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 30:
        result = _optimize(best_params, start_time)
        vx = result[:N]
        vy = result[N:2*N]
        vr = result[2*N:3*N]
        vx, vy, vr = _make_valid(vx, vy, vr)
        score = _sum_r(vr)
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx, vy, vr])
    
    # Phase 3: Tight polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 20:
        polished = _tight_polish(best_params, start_time)
        px = polished[:N]
        py = polished[N:2*N]
        pr = polished[2*N:3*N]
        px, py, pr = _make_valid(px, py, pr)
        p_score = _sum_r(pr)
        if p_score > best_score:
            best_score = p_score
            best_params = np.concatenate([px, py, pr])
    
    # Phase 4: Local search
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        ls_result = _local_search(best_params, start_time, max_iter=200)
        ls_x = ls_result[:N]
        ls_y = ls_result[N:2*N]
        ls_r = ls_result[2*N:3*N]
        ls_x, ls_y, ls_r = _make_valid(ls_x, ls_y, ls_r)
        ls_score = _sum_r(ls_r)
        if ls_score > best_score:
            best_score = ls_score
            best_params = np.concatenate([ls_x, ls_y, ls_r])
    
    # Phase 5: Final LP refinement on best centers
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 10:
        bx = best_params[:N]
        by = best_params[N:2*N]
        br = _optimize_radii_lp(bx, by)
        bx, by, br = _make_valid(bx, by, br)
        b_score = _sum_r(br)
        if b_score > best_score:
            best_score = b_score
            best_params = np.concatenate([bx, by, br])
    
    # Phase 6: Final tight polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 5:
        polished2 = _tight_polish(best_params, start_time)
        px2 = polished2[:N]
        py2 = polished2[N:2*N]
        pr2 = polished2[2*N:3*N]
        px2, py2, pr2 = _make_valid(px2, py2, pr2)
        p2_score = _sum_r(pr2)
        if p2_score > best_score:
            best_score = p2_score
            best_params = np.concatenate([px2, py2, pr2])
    
    if best_params is None:
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
