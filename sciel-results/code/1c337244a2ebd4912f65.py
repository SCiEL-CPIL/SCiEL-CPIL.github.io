"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# Precompute pair indices
_II, _JJ = np.triu_indices(N, k=1)


def _softplus(v, beta=50.0):
    """Smooth approximation of max(0, v) via softplus."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 20.0 / beta, v, np.log1p(np.exp(beta * v)) / beta)


def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[_II] + radii[_JJ]
    if np.any(dist_sq < (min_dist - tol) ** 2):
        return False
    return True


def _post_process(pos, radii):
    """
    Ensure strict validity by minimally adjusting radii/positions.
    Shrinks only the smaller circle in conflicts to preserve total radius.
    """
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        changed = False
        for i in range(N):
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound:
                radii[i] = max(0.0, max_r_bound) * (1.0 - 1e-10)
                changed = True

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    if radii[i] <= radii[j]:
                        radii[i] = max(0.0, dist - radii[j]) * (1.0 - 1e-10)
                    else:
                        radii[j] = max(0.0, dist - radii[i]) * (1.0 - 1e-10)
                    changed = True

        if not changed:
            break

    if not _is_valid(pos, radii, tol=1e-9):
        for i in range(N):
            radii[i] *= (1.0 - 1e-8)

    return pos, radii


def _max_radius_at(pos, radii, i):
    """Compute maximum radius for circle i given current positions of all circles."""
    max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
    for j in range(N):
        if j == i:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)


def _radius_expansion(pos, radii, max_rounds=200):
    """
    Dedicated radius expansion pass: fix positions, grow radii to fill gaps.
    Iteratively set each radius to its maximum possible value.
    """
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_rounds):
        changed = False
        for i in range(N):
            new_r = _max_radius_at(pos, radii, i)
            if new_r > radii[i] + 1e-12:
                radii[i] = new_r * (1.0 - 1e-10)
                changed = True
        if not changed:
            break

    return pos, radii


def _smooth_potential(pos, radii, beta=80.0):
    """
    Smooth distance-field potential energy.
    Minimizes sum of squared signed distances to boundary and pairwise contact surfaces.
    """
    # Boundary signed distances (positive = inside, negative = outside)
    bd_left = pos[:, 0] - radii
    bd_right = 1.0 - pos[:, 0] - radii
    bd_bottom = pos[:, 1] - radii
    bd_top = 1.0 - pos[:, 1] - radii

    # Smooth max(0, -d) for violations (d < 0 means violation)
    b_pen = (_softplus(-bd_left, beta) ** 2 +
             _softplus(-bd_right, beta) ** 2 +
             _softplus(-bd_bottom, beta) ** 2 +
             _softplus(-bd_top, beta) ** 2)

    # Pairwise: signed distance = dist - (r_i + r_j), positive = no overlap
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-20)
    signed_dist = dist - (radii[_II] + radii[_JJ])
    o_pen = _softplus(-signed_dist, beta) ** 2

    return np.sum(b_pen) + np.sum(o_pen)


def _objective_smooth(x, beta=80.0):
    """Objective for smooth optimization: -sum(radii) + smooth_potential."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    return -np.sum(radii) + 50.0 * _smooth_potential(pos, radii, beta)


def _slsqp_pass(pos, radii, deadline, ftol=1e-14, maxiter=200):
    """SLSQP pass with hard constraints."""
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    def constraint_boundary(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        c1 = p[:, 0] - r
        c2 = 1.0 - p[:, 0] - r
        c3 = p[:, 1] - r
        c4 = 1.0 - p[:, 1] - r
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        dx = p[_II, 0] - p[_JJ, 0]
        dy = p[_II, 1] - p[_JJ, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[_II] + r[_JJ]
        return dist_sq - min_dist * min_dist

    def objective(x):
        return -np.sum(x[2 * N:])

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-14, 'disp': False})
        pos_curr = res.x[:2 * N].reshape(N, 2)
        rad_curr = np.maximum(res.x[2 * N:], 1e-6)
        pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
        if _is_valid(pos_pp, rad_pp):
            return pos_pp, rad_pp
    except Exception:
        pass
    return pos, radii


def _best_fit_init(seed=0):
    """Greedy Best-Fit initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

        for _ in range(15000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if found and best_r > 0.01:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.98
        else:
            for _ in range(1000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                min_dist = min(x, 1 - x, y, 1 - y)
                for j in range(i):
                    d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                    min_dist = min(min_dist, d - radii[j])
                if min_dist > 0.005:
                    pos[i] = [x, y]
                    radii[i] = min_dist * 0.95
                    break
            else:
                pos[i] = [0.5, 0.5]
                radii[i] = 0.001

    return pos, radii


def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5

    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r

        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break

    rng = np.random.RandomState(123)
    while k < N:
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(k):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
        if best_r > 0.01:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.98
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        k += 1

    return pos, radii


def _corner_init():
    """Corner-biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # Place circles in corners first
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i, c in enumerate(corners):
        pos[i] = c
        radii[i] = 0.14

    rng = np.random.RandomState(456)
    for i in range(4, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(8000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
        if best_r > 0.01:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.98
        else:
            pos[i] = [0.5, 0.5]
            radii[i] = 0.01

    return pos, radii


def _local_search(pos, radii, deadline, max_iter=3000, step_scale=1.0):
    """Local search with random perturbations and simulated annealing."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_pos = pos.copy()
    current_radii = radii.copy()
    current_score = best_score

    rng = np.random.RandomState(42)
    temp = 0.015
    cooling = 0.9995

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        new_pos = current_pos.copy()
        new_radii = current_radii.copy()

        move_type = rng.randint(0, 4)

        if move_type == 0:
            n_perturb = rng.randint(1, 4)
            indices = rng.choice(N, n_perturb, replace=False)
            for idx in indices:
                new_pos[idx, 0] += rng.uniform(-0.012, 0.012) * step_scale
                new_pos[idx, 1] += rng.uniform(-0.012, 0.012) * step_scale
                new_radii[idx] *= (1 + rng.uniform(-0.03, 0.03) * step_scale)
                new_radii[idx] = max(new_radii[idx], 1e-6)
        elif move_type == 1:
            idx = rng.randint(N)
            corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                       [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]]
            target = corners[rng.randint(len(corners))]
            alpha = rng.uniform(0.1, 0.35)
            new_pos[idx, 0] = (1 - alpha) * new_pos[idx, 0] + alpha * target[0]
            new_pos[idx, 1] = (1 - alpha) * new_pos[idx, 1] + alpha * target[1]
            new_radii[idx] *= (1 + rng.uniform(-0.02, 0.04))
            new_radii[idx] = max(new_radii[idx], 1e-6)
        elif move_type == 2:
            i, j = rng.choice(N, 2, replace=False)
            new_pos[i, 0], new_pos[j, 0] = new_pos[j, 0], new_pos[i, 0]
            new_pos[i, 1], new_pos[j, 1] = new_pos[j, 1], new_pos[i, 1]
            new_pos[i, 0] += rng.uniform(-0.008, 0.008)
            new_pos[i, 1] += rng.uniform(-0.008, 0.008)
            new_pos[j, 0] += rng.uniform(-0.008, 0.008)
            new_pos[j, 1] += rng.uniform(-0.008, 0.008)
            new_radii[i] *= (1 + rng.uniform(-0.02, 0.02))
            new_radii[j] *= (1 + rng.uniform(-0.02, 0.02))
            new_radii[i] = max(new_radii[i], 1e-6)
            new_radii[j] = max(new_radii[j], 1e-6)
        else:
            idx = rng.randint(N)
            new_pos[idx, 0] += rng.uniform(-0.006, 0.006)
            new_pos[idx, 1] += rng.uniform(-0.006, 0.006)
            new_radii[idx] *= (1 + rng.uniform(-0.015, 0.025))
            new_radii[idx] = max(new_radii[idx], 1e-6)

        new_pos_pp, new_radii_pp = _post_process(new_pos, new_radii)
        if _is_valid(new_pos_pp, new_radii_pp):
            new_score = np.sum(new_radii_pp)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos_pp.copy()
                best_radii = new_radii_pp.copy()
            delta = new_score - current_score
            if new_score >= current_score or (delta < 0 and rng.uniform() < np.exp(delta / max(temp, 1e-10))):
                current_pos = new_pos_pp.copy()
                current_radii = new_radii_pp.copy()
                current_score = new_score

        temp *= cooling

    return best_pos, best_radii


def _smooth_optimize(pos, radii, deadline):
    """
    Smooth potential energy optimization using L-BFGS-B.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.0, 0.5)] * N

    # Stage 1: Moderate smoothness
    if time.time() < deadline - 20:
        try:
            res = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                           args=(50.0,),
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            pos_curr = res.x[:2 * N].reshape(N, 2)
            rad_curr = np.maximum(res.x[2 * N:], 1e-6)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
        except Exception:
            pass

    # Stage 2: Higher smoothness (sharper potential)
    if time.time() < deadline - 15:
        x0_2 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            res = minimize(_objective_smooth, x0_2, method='L-BFGS-B', bounds=bounds,
                           args=(100.0,),
                           options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'disp': False})
            pos_curr = res.x[:2 * N].reshape(N, 2)
            rad_curr = np.maximum(res.x[2 * N:], 1e-6)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
        except Exception:
            pass

    return best_pos, best_radii


def _final_refinement(pos, radii, deadline):
    """
    Multiple SLSQP passes with decreasing ftol plus smooth polish.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for ftol in [1e-12, 1e-14, 1e-16]:
        if time.time() > deadline - 5:
            break
        pos_new, rad_new = _slsqp_pass(best_pos, best_radii, deadline, ftol=ftol, maxiter=300)
        score = np.sum(rad_new)
        if score > best_score:
            best_score = score
            best_pos = pos_new.copy()
            best_radii = rad_new.copy()

    # Smooth polish
    if time.time() < deadline - 3:
        pos_s, rad_s = _smooth_optimize(best_pos, best_radii, deadline)
        score = np.sum(rad_s)
        if score > best_score:
            best_score = score
            best_pos = pos_s.copy()
            best_radii = rad_s.copy()

    # Final SLSQP
    if time.time() < deadline - 2:
        pos_new, rad_new = _slsqp_pass(best_pos, best_radii, deadline, ftol=1e-16, maxiter=400)
        score = np.sum(rad_new)
        if score > best_score:
            best_score = score
            best_pos = pos_new.copy()
            best_radii = rad_new.copy()

    return best_pos, best_radii


def _optimize(pos, radii, deadline, seed=0):
    """Multi-stage optimization pipeline."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    # Stage 1: Smooth potential optimization
    if time.time() < deadline - 30:
        pos_s, rad_s = _smooth_optimize(best_pos, best_radii, deadline)
        score = np.sum(rad_s)
        if score > best_score:
            best_score = score
            best_pos = pos_s.copy()
            best_radii = rad_s.copy()

    # Stage 2: SLSQP with hard constraints
    if time.time() < deadline - 25:
        pos_sl, rad_sl = _slsqp_pass(best_pos, best_radii, deadline, ftol=1e-12, maxiter=300)
        score = np.sum(rad_sl)
        if score > best_score:
            best_score = score
            best_pos = pos_sl.copy()
            best_radii = rad_sl.copy()

    # Stage 3: Radius expansion
    if time.time() < deadline - 20:
        pos_re, rad_re = _radius_expansion(best_pos, best_radii)
        pos_re, rad_re = _post_process(pos_re, rad_re)
        if _is_valid(pos_re, rad_re):
            score = np.sum(rad_re)
            if score > best_score:
                best_score = score
                best_pos = pos_re.copy()
                best_radii = rad_re.copy()

    # Stage 4: Local search
    if time.time() < deadline - 15:
        pos_ls, rad_ls = _local_search(best_pos, best_radii, deadline, max_iter=1500)
        if _is_valid(pos_ls, rad_ls):
            score = np.sum(rad_ls)
            if score > best_score:
                best_score = score
                best_pos = pos_ls.copy()
                best_radii = rad_ls.copy()

    # Stage 5: Final refinement
    if time.time() < deadline - 5:
        pos_f, rad_f = _final_refinement(best_pos, best_radii, deadline)
        score = np.sum(rad_f)
        if score > best_score:
            best_score = score
            best_pos = pos_f.copy()
            best_radii = rad_f.copy()

    # Stage 6: Final radius expansion + polish
    if time.time() < deadline - 2:
        pos_re2, rad_re2 = _radius_expansion(best_pos, best_radii)
        pos_re2, rad_re2 = _post_process(pos_re2, rad_re2)
        if _is_valid(pos_re2, rad_re2):
            score = np.sum(rad_re2)
            if score > best_score:
                best_score = score
                best_pos = pos_re2.copy()
                best_radii = rad_re2.copy()

    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_pos = None
    best_rad = None
    best_score = 0.0

    # Generate initializations
    inits = []

    pos_bf, rad_bf = _best_fit_init(seed=1)
    inits.append((pos_bf, rad_bf))

    pos_bf2, rad_bf2 = _best_fit_init(seed=7)
    inits.append((pos_bf2, rad_bf2))

    pos_hex, rad_hex = _hex_init()
    inits.append((pos_hex, rad_hex))

    pos_cor, rad_cor = _corner_init()
    inits.append((pos_cor, rad_cor))

    # Optimize each initialization
    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break

        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        if _is_valid(pos_pp, rad_pp):
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_rad = rad_pp.copy()

        try:
            pos_opt, rad_opt = _optimize(pos_pp, rad_pp, deadline, seed=i)
            pos_pp_opt, rad_pp_opt = _post_process(pos_opt, rad_opt)
            if _is_valid(pos_pp_opt, rad_pp_opt):
                score = np.sum(rad_pp_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp_opt.copy()
                    best_rad = rad_pp_opt.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    rad[k] = 0.095
                    k += 1
        best_pos, best_rad = _post_process(pos, rad)
        best_score = np.sum(best_rad)

    # Final global polish
    if time.time() < deadline - 8:
        pos_ls, rad_ls = _local_search(best_pos, best_rad, deadline, max_iter=2000)
        if _is_valid(pos_ls, rad_ls):
            score = np.sum(rad_ls)
            if score > best_score:
                best_score = score
                best_pos = pos_ls.copy()
                best_rad = rad_ls.copy()

    if time.time() < deadline - 5:
        pos_re, rad_re = _radius_expansion(best_pos, best_rad)
        pos_re, rad_re = _post_process(pos_re, rad_re)
        if _is_valid(pos_re, rad_re):
            score = np.sum(rad_re)
            if score > best_score:
                best_score = score
                best_pos = pos_re.copy()
                best_rad = rad_re.copy()

    if time.time() < deadline - 3:
        pos_f, rad_f = _final_refinement(best_pos, best_rad, deadline)
        score = np.sum(rad_f)
        if score > best_score:
            best_score = score
            best_pos = pos_f.copy()
            best_rad = rad_f.copy()

    best_pos, best_rad = _post_process(best_pos, best_rad)

    result = np.column_stack([best_pos, best_rad])
    return result

# EVOLVE-BLOCK-END
