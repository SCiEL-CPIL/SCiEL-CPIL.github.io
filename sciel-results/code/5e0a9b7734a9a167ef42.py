"""Circle packing in unit square - 26 circles maximizing sum of radii.
Keeps the compact structural parameterization family and adds:
  * a smoother, higher-beta soft-max penalty for the final polish
  * a dedicated high-precision radius-growth pass (binary-search each radius
    against the current contacts) to squeeze out more sum-of-radii
  * a tighter SLSQP cross-check with explicit constraints
  * re-tuned time budget to spend more on structural DE and final polish
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline, beta=60.0, quad_w=50.0, sm_w=20.0):
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        # guard against overflow
        shifted = np.clip(shifted, -500, 500)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    penalty = quad_w * quad_penalty + sm_w * sm
    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------

def _expand_struct(p):
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    pos[4] = [cx, cy]
    radii[4] = center_r

    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Post-processing ----------------

def _post_process(pos, radii, deadline):
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    return pos, radii


# ---------------- High-precision polish ----------------

def _polish(pos, radii, deadline, beta=200.0):
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    base_bounds = []
    for i in range(N):
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        base_bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        base_bounds.append((1e-4, 0.35))

    def obj_high_beta(x):
        return _objective(x, deadline, beta=beta, quad_w=100.0, sm_w=50.0)

    best_result = None
    best_score = -np.inf
    rng = np.random.RandomState(42)
    num_tries = 4
    for try_idx in range(num_tries):
        if time.time() > deadline - 5:
            break
        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.008, size=len(x0))
            x_try = x0 + perturbation
            x_try[2 * N:] = np.maximum(x_try[2 * N:], 1e-4)
            for i in range(N):
                x_try[2 * i] = np.clip(x_try[2 * i], 0.0, 1.0)
                x_try[2 * i + 1] = np.clip(x_try[2 * i + 1], 0.0, 1.0)
        try:
            result = minimize(obj_high_beta, x_try, method='L-BFGS-B',
                              bounds=base_bounds,
                              options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_result = (pos_pp, radii_pp)
        except Exception:
            pass

    if best_result is not None:
        return best_result
    return pos, radii


def _radius_grow(pos, radii, deadline):
    """Binary-search each radius upward while keeping the configuration valid.
    Iterate a few passes because growing one circle can constrain others."""
    pos = pos.copy()
    radii = radii.copy()

    def _can_grow(i, r_new):
        # boundary
        if r_new > pos[i, 0] or r_new > 1.0 - pos[i, 0]:
            return False
        if r_new > pos[i, 1] or r_new > 1.0 - pos[i, 1]:
            return False
        # overlaps
        for j in range(N):
            if j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r_new + radii[j] - 1e-12:
                return False
        return True

    for _pass in range(3):
        if time.time() > deadline - 2:
            break
        for i in range(N):
            if time.time() > deadline - 2:
                break
            lo = radii[i]
            hi = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if hi <= lo:
                continue
            # 40-step binary search
            for _ in range(40):
                if hi - lo < 1e-14:
                    break
                mid = 0.5 * (lo + hi)
                if _can_grow(i, mid):
                    lo = mid
                else:
                    hi = mid
            radii[i] = lo

    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.35)] * N

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i]) - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i + 1]) - x[2 * N + i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j:
                         np.sqrt((x[2 * i] - x[2 * j]) ** 2 +
                                 (x[2 * i + 1] - x[2 * j + 1]) ** 2)
                         - x[2 * N + i] - x[2 * N + j]})

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    best_result = None
    best_score = -np.inf
    rng = np.random.RandomState(123)
    num_tries = 2
    for try_idx in range(num_tries):
        if time.time() > deadline - 3:
            break
        if try_idx == 0:
            x_try = x0.copy()
        else:
            perturbation = rng.normal(0, 0.004, size=len(x0))
            x_try = x0 + perturbation
            x_try[2 * N:] = np.maximum(x_try[2 * N:], 1e-4)
            for i in range(N):
                x_try[2 * i] = np.clip(x_try[2 * i], 0.0, 1.0)
                x_try[2 * i + 1] = np.clip(x_try[2 * i + 1], 0.0, 1.0)
        try:
            result = minimize(neg_sum_rad, x_try, method='SLSQP', bounds=bounds,
                              constraints=cons,
                              options={'maxiter': 500, 'ftol': 1e-14})
            if result.x is not None:
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_result = (pos_pp, radii_pp)
        except Exception:
            pass

    if best_result is not None:
        return best_result
    return pos, radii


def _make_grid(n):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    struct_bounds = [
        (0.06, 0.28),
        (0.06, 0.32),
        (0.3, 0.7),
        (0.3, 0.7),
        (0.18, 0.45),
        (0.04, 0.16),
        (0.04, 0.16),
        (0.30, 0.55),
        (0.03, 0.12),
        (0.03, 0.12),
    ]

    # Track A: DE over structural space
    if time.time() < deadline - 90:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=200, popsize=32, seed=7,
                                            tol=1e-12, polish=False,
                                            mutation=(0.35, 1.0), recombination=0.75,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
        except Exception:
            pass

    if time.time() < deadline - 70:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=120, popsize=26, seed=123,
                                             tol=1e-11, polish=False,
                                             mutation=(0.5, 1.1), recombination=0.85,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
        except Exception:
            pass

    # L-BFGS-B on structural seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.105, 0.43, 0.065, 0.065],
        [0.135, 0.17, 0.5, 0.5, 0.32, 0.105, 0.105, 0.44, 0.065, 0.065],
        [0.125, 0.155, 0.5, 0.5, 0.315, 0.11, 0.11, 0.425, 0.07, 0.07],
        [0.13, 0.15, 0.5, 0.5, 0.31, 0.108, 0.108, 0.43, 0.068, 0.068],
        [0.14, 0.16, 0.5, 0.5, 0.305, 0.112, 0.112, 0.425, 0.072, 0.072],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # Track B: direct 78-dim from best structural seed
    if time.time() < deadline - 40 and best_pos is not None:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 800, 'ftol': 1e-14})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 15:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline, beta=300.0)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # Radius growth pass
    if best_pos is not None and time.time() < deadline - 8:
        pos_g, radii_g = _radius_grow(best_pos, best_radii, deadline)
        if _is_valid(pos_g, radii_g):
            s = np.sum(radii_g)
            if s > best_score:
                best_score = s
                best_pos = pos_g.copy()
                best_radii = radii_g.copy()

    # SLSQP cross-check
    if best_pos is not None and time.time() < deadline - 4:
        pos_q, radii_q = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_q, radii_q):
            s = np.sum(radii_q)
            if s > best_score:
                best_score = s
                best_pos = pos_q.copy()
                best_radii = radii_q.copy()

    # Another radius growth after SLSQP
    if best_pos is not None and time.time() < deadline - 1:
        pos_g2, radii_g2 = _radius_grow(best_pos, best_radii, deadline)
        if _is_valid(pos_g2, radii_g2):
            s = np.sum(radii_g2)
            if s > best_score:
                best_score = s
                best_pos = pos_g2.copy()
                best_radii = radii_g2.copy()

    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
