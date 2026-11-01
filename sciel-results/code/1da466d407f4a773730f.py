"""Packing 50 equal circles: smooth soft-min epigraph + L-BFGS + exact polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
T_KILL = 150.0

def _compute_r(cents):
    """Exact packing radius."""
    c = np.asarray(cents, dtype=np.float64)
    n = len(c)
    if n == 0:
        return 0.0
    # wall distance
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    wall = np.min(np.minimum(wx, wy))
    if n > 1:
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        pair = np.min(d) / 2.0
    else:
        pair = 0.5
    return float(min(wall, pair))

def _soft_min_obj_and_grad(t, cents, beta):
    """
    Objective: maximize t subject to soft-min distance >= 2t.
    We minimize:  0.5 * (2t - soft_min_dist)^2 + small*regularizer
    where soft_min_dist = -1/beta * log( sum_i exp(-beta * d_i) )
    d_i are the candidate distances (pairwise/2 and wall).
    Returns (obj, grad_t, grad_cents).
    """
    c = np.asarray(cents, dtype=np.float64)
    n = len(c)
    # gather all distances (pairwise/2 and wall per point)
    # pairwise
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_d = d / 2.0
    pair_d = pair_d[pair_d < np.inf]
    # wall
    wall = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    all_d = np.concatenate([pair_d, wall])
    # soft min
    # numerical safety
    all_d = np.clip(all_d, 1e-12, None)
    m = np.min(all_d)
    # log-sum-exp shift
    shifted = -beta * (all_d - m)
    # clip for stability
    shifted = np.clip(shifted, -50, 50)
    logsum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - logsum / beta
    # target
    gap = 2.0 * t - soft_min
    obj = 0.5 * gap * gap
    # gradients
    # d soft_min / d d_i = exp(-beta*(d_i - m)) / sum_j exp(-beta*(d_j - m))
    exp_shift = np.exp(shifted)
    w = exp_shift / np.sum(exp_shift)  # weights sum to 1
    # d soft_min / d t = 0
    # grad_t = 2*gap * ( - d soft_min/dt ) = 2*gap * 0 = 0? 
    # Wait: obj = 0.5*(2t - soft_min)^2, d obj/dt = (2t - soft_min)*2 = 2*gap
    grad_t = 2.0 * gap
    # grad_cents: d obj/d c = 2*gap * (- d soft_min/d c) = -2*gap * d soft_min/d c
    # d soft_min/d c = - sum_i w_i * d d_i/d c
    # d pair_d/d c for pair (i,j): pair_d = |c_i - c_j|/2, d/d c_i = (c_i - c_j)/|c_i-c_j| * 1 ... 
    # Let's compute d soft_min / d c via chain: for each distance term, weight w * grad of that distance.
    # Build gradient accumulator
    grad_c = np.zeros_like(c)
    # pairwise contributions
    # d = |c_i - c_j|, pair_d = d/2, d pair_d/d c_i = (c_i - c_j)/ (2 * d) * ... actually d|c_i-c_j|/d c_i = (c_i-c_j)/|c_i-c_j|
    # so d pair_d/d c_i = (c_i - c_j)/(2*|c_i-c_j|)
    # We have w over all_d; need to map back. Let's recompute with structure.
    # Rebuild structured weights
    # pairwise matrix
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    pairD = D / 2.0
    # clip
    pairD_safe = np.clip(pairD, 1e-12, None)
    # weights for pairwise: w_pair[i,j] = exp(-beta*(pairD[i,j]-m))/Z
    # wall: w_wall[i] = exp(-beta*(wall[i]-m))/Z
    # Z = sum of all exp terms
    exp_pair = np.exp(-beta * (pairD_safe - m))
    exp_wall = np.exp(-beta * (wall - m))
    Z = np.sum(exp_pair) + np.sum(exp_wall)
    exp_pair = np.clip(exp_pair, 0, None)
    exp_wall = np.clip(exp_wall, 0, None)
    Z = max(Z, 1e-30)
    w_pair = exp_pair / Z
    w_wall = exp_wall / Z
    # d soft_min/d c_i = - ( sum_{j} w_pair[i,j] * d pairD[i,j]/d c_i + w_wall[i] * d wall[i]/d c_i )
    # d pairD[i,j]/d c_i = (c_i - c_j)/(2 * D[i,j])
    # d wall[i]/d c_i: wall[i] = min(0.5-|x_i|, 0.5-|y_i|) -> subgradient; use sign
    # For smoothness use: d(0.5-|x|)/dx = -sign(x), d(0.5-|y|)/dy = -sign(y)
    # But wall is min of two; approximate with the active one or average. Use gradient of the min via soft? 
    # Simpler: treat wall contribution as gradient of 0.5 - |x| and 0.5 - |y| weighted by which is smaller.
    # For now use: grad_wall_x = -sign(x_i) if wx<wy else 0; grad_wall_y similarly. To keep smooth, use soft.
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    # soft min of wx, wy per point
    b2 = 50.0
    mm = np.minimum(wx, wy)
    sw = np.exp(-b2 * (wx - mm)); swy = np.exp(-b2 * (wy - mm))
    zz = sw + swy + 1e-30
    gw_x = -np.sign(c[:, 0]) * sw / zz
    gw_y = -np.sign(c[:, 1]) * swy / zz
    # pairwise grad
    # D with inf diagonal; avoid div by inf
    D_safe = np.where(D < np.inf, D, 1.0)
    unit = diff / D_safe[:, :, None]  # (i-j)/|i-j|
    # d pairD[i,j]/d c_i = unit[i,j] / 2
    # accumulate: for each i, sum_j w_pair[i,j] * unit[i,j]/2
    # w_pair is symmetric; use upper triangle or full
    # grad_c[i] += -2*gap * ( sum_j w_pair[i,j]*unit[i,j]/2 + w_wall[i]*[gw_x[i], gw_y[i]] )
    # Let's compute sum_j w_pair[i,j]*unit[i,j]
    # w_pair[i,j] * unit[i,j] -> contract over j
    # unit[i,j] = (c_i - c_j)/D
    # sum_j w_pair[i,j]*(c_i - c_j)/D[i,j]
    # = c_i * sum_j w_pair[i,j]/D[i,j] - sum_j w_pair[i,j]*c_j/D[i,j]
    invD = np.where(D < np.inf, 1.0 / D_safe, 0.0)
    # w_pair * invD
    winv = w_pair * invD
    sum_winv = np.sum(winv, axis=1)  # per i
    # sum_j w_pair[i,j]*c_j / D[i,j]
    # (winv @ c)
    term = winv @ c  # shape (n,2)
    pair_grad = c * sum_winv[:, None] - term
    # d soft_min/d c_i = - ( pair_grad[i]/2 + w_wall[i]*[gw_x[i], gw_y[i]] )
    # d obj/d c_i = -2*gap * d soft_min/d c_i = 2*gap*(pair_grad[i]/2 + w_wall[i]*gw)
    grad_c = 2.0 * gap * (pair_grad / 2.0 + w_wall[:, None] * np.column_stack([gw_x, gw_y]))
    return obj, grad_t, grad_c

def _init_hex():
    """Hexagonal initial arrangement."""
    best_cents = None
    best_r = 0.0
    for n_rows in [5, 6, 7, 8, 9, 10]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        if r <= best_r:
            continue
        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        centers = np.array(centers[:N])
        # clamp
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
        r_actual = _compute_r(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _compute_r(best_cents)
    return best_cents, best_r

def _init_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c = grid[:N]
    return c, _compute_r(c)

def _init_random(seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.45, 0.45, size=(N, 2))
    return c, _compute_r(c)

def _lbfgs_optimize(cents0, t0, time_budget):
    """Run L-BFGS on the soft-min epigraph objective."""
    start = time.time()
    beta = 80.0  # softness
    def obj_grad(x):
        t = x[0]
        c = x[1:].reshape(N, 2)
        o, gt, gc = _soft_min_obj_and_grad(t, c, beta)
        g = np.concatenate([[gt], gc.ravel()])
        return o, g
    x0 = np.concatenate([[t0], cents0.ravel()])
    res = minimize(obj_grad, x0, method='L-BFGS-B',
                   options={'maxiter': 2000, 'maxfun': 2000,
                            'ftol': 1e-14, 'gtol': 1e-8,
                            'maxls': 50})
    t_opt = res.x[0]
    c_opt = res.x[1:].reshape(N, 2)
    # clamp to square
    c_opt = np.clip(c_opt, -0.5 + 1e-12, 0.5 - 1e-12)
    return c_opt, t_opt

def _exact_local_polish(cents, time_budget):
    """High-precision local polish: small-step exact-min hill climbing with adaptive step."""
    start = time.time()
    best = cents.copy()
    best_r = _compute_r(best)
    rng = np.random.default_rng(123)
    step = 0.002
    n = len(best)
    while time.time() - start < time_budget:
        # try moving one point
        i = rng.integers(n)
        for _ in range(40):
            if time.time() - start >= time_budget:
                break
            delta = rng.normal(0, step, size=2)
            new_c = best.copy()
            new_c[i] = best[i] + delta
            if np.any(np.abs(new_c) > 0.5 - 1e-12):
                continue
            r = _compute_r(new_c)
            if r > best_r:
                best_r = r
                best = new_c
                step = max(step * 0.97, 1e-5)
            else:
                step = max(step * 0.995, 1e-5)
    return best, best_r

def _refine_radius(cents):
    """Binary search on radius to confirm feasibility / find max t for this config."""
    r = _compute_r(cents)
    return r

def construct():
    start = time.time()
    deadline = T_KILL - 15.0  # safety margin

    # Multiple initializations
    inits = []
    c1, r1 = _init_hex()
    inits.append((c1, r1))
    c2, r2 = _init_grid()
    inits.append((c2, r2))
    for s in [1, 7, 42]:
        cr, rr = _init_random(s)
        inits.append((cr, rr))

    best_cents = inits[0][0].copy()
    best_r = inits[0][1]
    for c, r in inits:
        if r > best_r:
            best_r = r
            best_cents = c.copy()

    # L-BFGS soft-min optimization from best init (and a couple others)
    t0 = best_r
    used = 0.0
    candidates = []
    for idx, (c, r) in enumerate(inits[:3]):  # top 3 inits
        if time.time() - start > deadline - 20:
            break
        t0_i = r
        try:
            c_opt, t_opt = _lbfgs_optimize(c, t0_i, time_budget=30)
            r_opt = _compute_r(c_opt)
            candidates.append((c_opt, r_opt))
            if r_opt > best_r:
                best_r = r_opt
                best_cents = c_opt.copy()
        except Exception:
            pass

    # Pick best candidate for polish
    if candidates:
        best_cand = max(candidates, key=lambda x: x[1])
        if best_cand[1] > best_r:
            best_r = best_cand[1]
            best_cents = best_cand[0].copy()

    # Exact local polish with remaining time
    remaining = deadline - (time.time() - start)
    if remaining > 5:
        c_pol, r_pol = _exact_local_polish(best_cents, time_budget=remaining - 3)
        if r_pol > best_r:
            best_r = r_pol
            best_cents = c_pol.copy()

    # Final clamp
    best_cents = np.clip(best_cents, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_cents
