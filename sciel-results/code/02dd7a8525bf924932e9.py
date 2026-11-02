import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 138.0  # evaluator kills at 150s; keep 12s margin

_global_start = None

def _elapsed(start=None):
    s = _global_start if start is None else start
    return time.time() - s

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

def _optimize_radii_lp(x, y):
    """Maximize sum(r) for fixed centers via LP: r_i+r_j<=d_ij, r_i<=dist_to_bound."""
    if _elapsed() > TIME_LIMIT - 2:
        return None
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    for i in range(N):
        b = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
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
    # Fallback: iterative shrink
    r = np.array([min(x[i], 1.0 - x[i], y[i], 1.0 - y[i]) for i in range(N)])
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if r[i] + r[j] > d:
                    shrink = (d / (r[i] + r[j])) * 0.999 if (r[i] + r[j]) > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return r

def _make_valid(x, y, r, max_iter=200):
    x = x.copy(); y = y.copy(); r = r.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            r[i] = min(r[i], x[i] * 0.999, (1.0 - x[i]) * 0.999,
                       y[i] * 0.999, (1.0 - y[i]) * 0.999)
            if r[i] < 1e-6:
                r[i] = 1e-6
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    shrink = (d / min_d) * 0.999 if min_d > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _obj_and_grad(params, soft_eps, mu):
    """Penalized objective + analytic gradient. Returns (value, grad)."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r_safe = np.maximum(r, 1e-10)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-14)
    dist[dist < 1e-14] = 1e-14
    min_dist = r_safe[:, None] + r_safe[None, :]
    diff = min_dist - dist

    eps = soft_eps * soft_eps
    sqrt_term = np.sqrt(diff * diff + eps)
    sqrt_term[sqrt_term < 1e-14] = 1e-14
    dsmooth = 0.5 * (1 + diff / sqrt_term)

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    obj = -np.sum(r_safe)
    overlap_pen = np.sum(dsmooth[mask])

    dneg_x = r_safe - x
    dpos_x = x - (1.0 - r_safe)
    dneg_y = r_safe - y
    dpos_y = y - (1.0 - r_safe)

    def sdrv(v):
        sv = np.sqrt(v * v + eps)
        sv[sv < 1e-14] = 1e-14
        return 0.5 * (1 + v / sv)

    sm_dn_x = sdrv(dneg_x)
    sm_dp_x = sdrv(dpos_x)
    sm_dn_y = sdrv(dneg_y)
    sm_dp_y = sdrv(dpos_y)

    bound_pen = np.sum(0.5 * (dneg_x + np.sqrt(dneg_x*dneg_x + eps)) +
                       0.5 * (dpos_x + np.sqrt(dpos_x*dpos_x + eps)) +
                       0.5 * (dneg_y + np.sqrt(dneg_y*dneg_y + eps)) +
                       0.5 * (dpos_y + np.sqrt(dpos_y*dpos_y + eps)))

    obj_val = obj + mu * (overlap_pen + bound_pen)

    grad = np.zeros(3 * N)
    # Pairwise overlap contributions
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                ds = dsmooth[i, j] * mu
                grad[i] += -ds * ux
                grad[N + i] += -ds * uy
                grad[j] += ds * ux
                grad[N + j] += ds * uy
                grad[2*N + i] += ds
                grad[2*N + j] += ds
    # Boundary + objective gradients
    for i in range(N):
        grad[i] += mu * (-sm_dn_x[i] + sm_dp_x[i])
        grad[N + i] += mu * (-sm_dn_y[i] + sm_dp_y[i])
        grad[2*N + i] += -1.0 + mu * (sm_dn_x[i] - sm_dp_x[i] + sm_dn_y[i] - sm_dp_y[i])

    return obj_val, grad

def _objective(params, start, soft_eps=1e-4, mu=50.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    return _obj_and_grad(params, soft_eps, mu)[0]

def _objective_grad(params, start, soft_eps=1e-4, mu=50.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros(3 * N)
    return _obj_and_grad(params, soft_eps, mu)

def _objective_tight_grad(params, start, soft_eps=1e-6, mu=200.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros(3 * N)
    return _obj_and_grad(params, soft_eps, mu)

def _push_apart(x, y, r, iterations=50):
    x = x.copy(); y = y.copy()
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
                    nx, ny = dx / d, dy / d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        for i in range(N):
            x[i] = np.clip(x[i], r[i] * 0.99, 1.0 - r[i] * 0.99)
            y[i] = np.clip(y[i], r[i] * 0.99, 1.0 - r[i] * 0.99)
        if not moved:
            break
    return x, y

def _solve(params, start, soft_eps, mu, maxiter, ftol, gtol, maxfun):
    """Run L-BFGS-B with analytic gradient; returns (params, improved)."""
    try:
        res = minimize(_objective_grad, params, jac=True, method='L-BFGS-B',
                       args=(start, soft_eps, mu),
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxfun': maxfun})
        if res.success or (hasattr(res, 'x') and res.x is not None):
            return res.x, True
    except Exception:
        pass
    return params, False

def _optimize(params, start, soft_eps=1e-4, mu=50.0, maxiter=3000,
              ftol=1e-14, gtol=1e-12, maxfun=50000):
    return _solve(params, start, soft_eps, mu, maxiter, ftol, gtol, maxfun)[0]

def _tight_polish(params, start, soft_eps=1e-6, mu=200.0, maxiter=5000,
                  ftol=1e-15, gtol=1e-14, maxfun=100000):
    return _solve(params, start, soft_eps, mu, maxiter, ftol, gtol, maxfun)[0]

def _refine_loop(params, start, max_rounds=6):
    """Iterate: optimize centers -> LP radii -> reoptimize. Drives to tighter fixed point."""
    best = params.copy()
    best_score = -_sum_r(best[2*N:3*N])
    for _ in range(max_rounds):
        if _elapsed(start) > TIME_LIMIT - 20:
            break
        # optimize centers/radii
        p, _ = _solve(best, start, 1e-4, 50.0, 1500, 1e-14, 1e-12, 25000)
        vx, vy = p[:N], p[N:2*N]
        vr = _optimize_radii_lp(vx, vy)
        if vr is None:
            break
        if not _is_valid(vx, vy, vr):
            vx, vy, vr = _make_valid(vx, vy, vr)
        sc = _sum_r(vr)
        if sc > best_score:
            best_score = sc
            best = np.concatenate([vx, vy, vr])
        else:
            # no improvement; tighten centers with the current LP radii as target
            p2, _ = _solve(best, start, 1e-6, 200.0, 1500, 1e-15, 1e-14, 25000)
            vx2, vy2 = p2[:N], p2[N:2*N]
            vr2 = _optimize_radii_lp(vx2, vy2)
            if vr2 is not None:
                if not _is_valid(vx2, vy2, vr2):
                    vx2, vy2, vr2 = _make_valid(vx2, vy2, vr2)
                sc2 = _sum_r(vr2)
                if sc2 > best_score:
                    best_score = sc2
                    best = np.concatenate([vx2, vy2, vr2])
    return best

def _local_search(params, start, max_iter=300):
    """Local search with random perturbations to escape local minima (gradient-accelerated)."""
    best_params = params.copy()
    best_score = _sum_r(best_params[2*N:3*N])
    rng = np.random.RandomState(12345)
    for i in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 12:
            break
        scale = 0.05 * (1.0 - i / max_iter)
        perturbed = best_params + rng.normal(0, scale, size=3 * N)
        px = perturbed[:N].copy()
        py = perturbed[N:2*N].copy()
        pr = perturbed[2*N:3*N].copy()
        px, py = _push_apart(px, py, pr, iterations=20)
        p_opt = np.concatenate([px, py, pr])
        p_opt, _ = _solve(p_opt, start, 1e-4, 50.0, 600, 1e-12, 1e-10, 6000)
        vx, vy = p_opt[:N], p_opt[N:2*N]
        vr = _optimize_radii_lp(vx, vy)
        if vr is None:
            continue
        if not _is_valid(vx, vy, vr):
            vx, vy, vr = _make_valid(vx, vy, vr)
        score = _sum_r(vr)
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx, vy, vr])
    return best_params

def _hex_init():
    rows = [6, 5, 6, 5, 6, 4]
    x = []; y = []; r = []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.5 / count if row_idx % 2 == 1 else 0.0
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            cr = 0.48 * row_h
            cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
            x.append(cx); y.append(ry); r.append(cr)
    x = x[:N]; y = y[:N]; r = r[:N]
    while len(x) < N:
        x.append(0.5); y.append(0.5); r.append(0.02)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _rect_init():
    side = int(np.ceil(np.sqrt(N)))
    x = []; y = []; r = []
    for i in range(side):
        for j in range(side):
            if len(x) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                cr = 0.48 / side
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                x.append(cx); y.append(cy); r.append(cr)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return x, y, r

def construct():
    global _global_start
    _global_start = time.time()
    start_time = _global_start

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
        hx2 = np.clip(hx2, 0, 1); hy2 = np.clip(hy2, 0, 1)
        inits.append(np.concatenate([hx2, hy2, hr2]))

    best_params = None
    best_score = -np.inf

    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 35:
            break
        x0a = x0[:N].copy(); x0b = x0[N:2*N].copy(); x0c = x0[2*N:3*N].copy()
        x0a, x0b = _push_apart(x0a, x0b, x0c, iterations=30)
        x0_opt = np.concatenate([x0a, x0b, x0c])
        result = _optimize(x0_opt, start_time)
        vx, vy = result[:N], result[N:2*N]
        vr = _optimize_radii_lp(vx, vy)
        if vr is None:
            vr = x0c.copy()
        if not _is_valid(vx, vy, vr):
            vx, vy, vr = _make_valid(vx, vy, vr)
        score = _sum_r(vr)
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx, vy, vr])

    # Refine the best via the LP-driven fixed-point loop
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 25:
        rp = _refine_loop(best_params, start_time, max_rounds=6)
        s = _sum_r(rp[2*N:3*N])
        if s > best_score:
            best_score = s
            best_params = rp

    # Local search from best result (gradient-accelerated)
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        ls_result = _local_search(best_params, start_time, max_iter=300)
        ls_x, ls_y = ls_result[:N], ls_result[N:2*N]
        ls_r = _optimize_radii_lp(ls_x, ls_y)
        if ls_r is None:
            ls_r = ls_result[2*N:3*N].copy()
        if not _is_valid(ls_x, ls_y, ls_r):
            ls_x, ls_y, ls_r = _make_valid(ls_x, ls_y, ls_r)
        ls_score = _sum_r(ls_r)
        if ls_score > best_score:
            best_score = ls_score
            best_params = np.concatenate([ls_x, ls_y, ls_r])

    # Final tight polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 8:
        polished = _tight_polish(best_params, start_time)
        px, py = polished[:N], polished[N:2*N]
        pr = _optimize_radii_lp(px, py)
        if pr is None:
            pr = polished[2*N:3*N].copy()
        if not _is_valid(px, py, pr):
            px, py, pr = _make_valid(px, py, pr)
        p_score = _sum_r(pr)
        if p_score > best_score:
            best_score = p_score
            best_params = np.concatenate([px, py, pr])

    if best_params is None:
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    return np.column_stack((x, y, r))
