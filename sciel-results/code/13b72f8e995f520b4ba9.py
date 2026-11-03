"""Combined strategy: Beta-continuation log-sum-exp surrogate with L-BFGS-B,
Nesterov active-set refinement, and diverse initializations for 50 equal spheres
in [-1/2, 1/2]^3."""
import time
import numpy as np
import scipy.optimize as opt
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill


def _score(pts):
    """Exact radius: min(min_pair_dist/2, min_wall_dist)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3) or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() * 0.5
    min_wall = np.min(0.5 - np.abs(pts))
    return float(min(min_pair, min_wall))


def _soft_min_obj_grad(x, beta):
    """
    Smooth surrogate: -soft_min of all radius-candidates (pairwise half-distances + wall distances).
    Returns (value, gradient) for minimization.
    """
    c = x.reshape(N, 3)

    # Wall clearances: 0.5 - |c_ik| for each coordinate
    wall = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall.ravel()  # shape (3N,)

    # Pairwise distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]  # shape (N*(N-1)/2,)

    # Combine all constraints
    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)

    # Soft-min via log-sum-exp
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta

    # Weights for each constraint in the soft-min
    weights = exp_vals / sum_exp  # shape (n_vals,)

    # Gradient w.r.t. x
    grad_x = np.zeros_like(c)  # shape (N, 3)

    # Wall contribution: dv/dx = -sign(c) for each coordinate
    wall_weights = weights[:3 * N]
    wall_weights_reshaped = wall_weights.reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= wall_weights_reshaped * sign_c

    # Pair contribution
    pair_weights = weights[3 * N:]

    ci = c[i]
    cj = c[j]
    diff = ci - cj
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)

    grad_ci = diff / (2.0 * norms)
    grad_cj = -grad_ci

    weighted_grad_ci = pair_weights[:, np.newaxis] * grad_ci
    weighted_grad_cj = pair_weights[:, np.newaxis] * grad_cj

    np.add.at(grad_x, i, weighted_grad_ci)
    np.add.at(grad_x, j, weighted_grad_cj)

    # Objective is -soft_min
    f = -soft_min_val
    g = -grad_x.ravel()

    return f, g


def _nesterov_active_set(c_init, maxiter=200, lr_start=0.002, lr_end=0.0001, deadline=DEADLINE_SECONDS):
    """
    Nesterov-accelerated gradient ascent on the exact min-distance objective.
    Uses active-set (k-NN) subgradient approximation for efficiency.
    """
    c = c_init.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _score(c)

    for it in range(maxiter):
        if time.time() > deadline - 3:
            break

        lr = lr_start * (lr_end / lr_start) ** (it / max(1, maxiter))

        # Compute active-set subgradient
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        # Wall distances
        wall_d = np.min(0.5 - np.abs(c), axis=1)

        current_min = min(np.min(d) / 2.0, np.min(wall_d))
        threshold = max(1.2 * current_min, 0.01)

        # Pairwise active set
        active_pair_mask = (d < 2.0 * threshold)
        active_pair_mask = active_pair_mask & (np.triu(np.ones((N, N), dtype=bool), 1))
        ai, aj = np.where(active_pair_mask)

        grad = np.zeros_like(c)
        if len(ai) > 0:
            d_active = d[ai, aj]
            d_safe = np.maximum(d_active, 1e-12)
            unit_diffs = (c[ai] - c[aj]) / d_safe[:, None]
            np.add.at(grad, ai, unit_diffs)
            np.add.at(grad, aj, -unit_diffs)

        # Wall active set
        for i in range(N):
            for dim in range(3):
                if c[i, dim] > 0:
                    if 0.5 - c[i, dim] < threshold:
                        grad[i, dim] -= 1.0
                else:
                    if c[i, dim] + 0.5 < threshold:
                        grad[i, dim] += 1.0

        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm

        # Nesterov update
        v = 0.9 * v + lr * grad
        c_new = c + v
        c_new = np.clip(c_new, -0.499999, 0.499999)

        # Check improvement
        r_new = _score(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        c = c_new

    return best_c


def _lbgfsb_beta_stage(pts, beta, maxiter, ftol, gtol, deadline):
    """Run one L-BFGS-B stage at given beta, respecting deadline."""
    x0 = pts.ravel()

    def fun(x):
        val, _ = _soft_min_obj_grad(x, beta)
        return val

    def jac(x):
        _, grad = _soft_min_obj_grad(x, beta)
        return grad

    best_x = x0.copy()
    best_f = fun(x0)

    def cb(xk):
        nonlocal best_x, best_f
        f = fun(xk)
        if f < best_f:
            best_f = f
            best_x = xk.copy()
        if time.time() > deadline - 5:
            raise RuntimeError("timeout")

    try:
        res = opt.minimize(fun, x0, jac=jac, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': maxiter, 'maxfun': maxiter * 4,
                                    'ftol': ftol, 'gtol': gtol, 'maxls': 50,
                                    'callback': cb})
        if res.x is not None:
            f = fun(res.x)
            if f < best_f:
                best_f = f
                best_x = res.x.copy()
    except RuntimeError:
        pass
    except Exception:
        pass

    result = best_x.reshape(N, 3)
    result = np.clip(result, -0.499999, 0.499999)
    return result


def _make_starts(rng):
    """Generate diverse initial configurations."""
    starts = []

    # 4x4x4 grid (first 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    starts.append(grid[:N].copy())

    # 5x5x2 slab
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    if len(slab) >= N:
        starts.append(slab[:N].copy())

    # Spectral/rotated initializations: random orthogonal rotations of grid
    base_grid = starts[0].copy()
    for _ in range(4):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        rotated = base_grid @ Q.T
        max_abs = np.max(np.abs(rotated))
        if max_abs > 0:
            rotated = rotated * (0.45 / max_abs)
        rotated = np.clip(rotated, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(rotated)

    # BCC lattice
    positions = []
    a = 0.5
    spacing = (2 * a) / 3.0
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                x = -a + (ix + 0.5) * spacing
                y = -a + (iy + 0.5) * spacing
                z = -a + (iz + 0.5) * spacing
                positions.append([x, y, z])
                positions.append([x + spacing / 2, y + spacing / 2, z + spacing / 2])
    bcc = np.array(positions[:N])
    bcc = np.clip(bcc, -0.5 + 1e-6, 0.5 - 1e-6)
    starts.append(bcc)

    # Random jittered grids
    for _ in range(3):
        s = starts[0] + rng.normal(0.0, 0.03, size=starts[0].shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)

    return starts


def _anneal(pts, T0, T1, steps, rng, deadline):
    """Simulated annealing with adaptive kicks."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.015 * (T / T0 + 0.05)
        n_move = max(1, N // 4)
        idx = rng.choice(N, size=n_move, replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand = np.clip(cand, -0.5 + 1e-12, 0.5 - 1e-12)
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 200 == 0 and time.time() > deadline - 5:
            break
    return best, best_r


def _beta_continuation(c_init, deadline):
    """
    Run beta-continuation optimization: gradually increase beta to sharpen
    the log-sum-exp surrogate, allowing the optimizer to navigate the landscape.
    """
    best_c = c_init.copy()
    best_val = _compute_radius_safe(best_c)

    # Continuation schedule: start smooth, end sharp
    beta_schedule = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    # Corresponding tolerances: tighten as beta increases
    tol_schedule = [
        (1e-10, 1e-6),
        (1e-10, 1e-6),
        (1e-11, 1e-7),
        (1e-11, 1e-7),
        (1e-12, 1e-8),
        (1e-12, 1e-8),
        (1e-13, 1e-9),
        (1e-13, 1e-9),
        (1e-14, 1e-10),
    ]

    for idx, beta in enumerate(beta_schedule):
        if time.time() > deadline - 15:
            break

        ftol, gtol = tol_schedule[idx]
        maxiter = 300 if beta <= 160 else 400

        new_c = _lbgfsb_beta_stage(best_c, beta, maxiter, ftol, gtol, deadline)
        new_val = _compute_radius_safe(new_c)

        if new_val > best_val:
            best_val = new_val
            best_c = new_c.copy()

    # Final high-precision polish with very high beta
    for beta_polish in [5000, 10000, 20000]:
        if time.time() > deadline - 8:
            break

        new_c = _lbgfsb_beta_stage(best_c, beta_polish, 300, 1e-15, 1e-11, deadline)
        new_val = _compute_radius_safe(new_c)

        if new_val > best_val:
            best_val = new_val
            best_c = new_c.copy()

    return best_c, best_val


def _compute_radius_safe(c):
    """Safe radius computation."""
    try:
        return _score(c)
    except Exception:
        return 0.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = np.random.default_rng(777)

    starts = _make_starts(rng)

    best = starts[0].copy()
    best_r = _score(best)

    # Phase 1: For each diverse start, do anneal + Nesterov + beta continuation
    for s in starts:
        if time.time() > deadline - 40:
            break

        # Quick anneal to escape obvious bad regions
        a, ar = _anneal(s, T0=0.03, T1=0.0005, steps=1500, rng=rng, deadline=deadline)
        if ar > best_r:
            best_r, best = ar, a

        # Nesterov active-set refinement
        if time.time() > deadline - 30:
            break
        n = _nesterov_active_set(a, maxiter=200, lr_start=0.002, lr_end=0.0001, deadline=deadline)
        nr = _score(n)
        if nr > best_r:
            best_r, best = nr, n

        # Beta continuation from the Nesterov result
        if time.time() > deadline - 20:
            break
        p, pr = _beta_continuation(n, deadline)
        if pr > best_r:
            best_r, best = pr, p

    # Phase 2: Additional beta continuation on the global best
    if time.time() < deadline - 15:
        p, pr = _beta_continuation(best, deadline)
        if pr > best_r:
            best_r, best = pr, p

    # Phase 3: Final Nesterov polish
    if time.time() < deadline - 8:
        n = _nesterov_active_set(best, maxiter=150, lr_start=0.001, lr_end=0.00005, deadline=deadline)
        nr = _score(n)
        if nr > best_r:
            best_r, best = nr, n

    # Phase 4: Ultra-fine beta polish if time remains
    if time.time() < deadline - 5:
        for beta in [30000, 50000]:
            if time.time() > deadline - 3:
                break
            p = _lbgfsb_beta_stage(best, beta, 200, 1e-15, 1e-12, deadline)
            pr = _score(p)
            if pr > best_r:
                best_r, best = pr, p

    # Final safety: ensure strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)

    return best
