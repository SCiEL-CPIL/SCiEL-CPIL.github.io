"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using log-sum-exp."""
    beta = max(beta, 1e-3)
    return (1.0 / beta) * np.log(np.mean(np.exp(beta * v)) + np.exp(-beta * 10.0))

def _penalty_function(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    Uses soft-max for smoothness.
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-8)

    # Boundary violations (smooth)
    bv_left = _soft_max(radii - pos[:, 0], beta)
    bv_right = _soft_max(pos[:, 0] + radii - 1.0, beta)
    bv_bottom = _soft_max(radii - pos[:, 1], beta)
    bv_top = _soft_max(pos[:, 1] + radii - 1.0, beta)
    boundary_pen = np.sum(bv_left**2 + bv_right**2 + bv_bottom**2 + bv_top**2)

    # Overlap violations (smooth)
    ii, jj = np.triu_indices(N, k=1)
    dx = pos[ii, 0] - pos[jj, 0]
    dy = pos[ii, 1] - pos[jj, 1]
    dist_sq = np.maximum(dx * dx + dy * dy, 1e-12)
    min_dist = radii[ii] + radii[jj]
    viol = _soft_max(min_dist - np.sqrt(dist_sq), beta)
    overlap_pen = np.sum(viol**2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)


def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    ii, jj = np.triu_indices(N, k=1)
    dx = pos[ii, 0] - pos[jj, 0]
    dy = pos[ii, 1] - pos[jj, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[ii] + radii[jj]
    if np.any(dist_sq < (min_dist - tol) ** 2):
        return False
    return True


def _post_process(pos, radii, shrink_factor=0.9999):
    """
    Ensure strict validity by minimally adjusting radii/positions.
    """
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(200):
        changed = False
        for i in range(N):
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound:
                radii[i] = max_r_bound * (1.0 - 1e-10)
                changed = True

            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale = dist / total_r * (1.0 - 1e-10)
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True

        if not changed:
            break

    if not _is_valid(pos, radii, tol=1e-9):
        for i in range(N):
            radii[i] *= (1.0 - 1e-8)

    return pos, radii


def _radius_growth_pass(pos, radii, deadline):
    """
    Fix positions and maximize radii via SLSQP with hard constraints.
    This squeezes additional sum-of-radii by growing circles into gaps.
    """
    if time.time() > deadline - 3:
        return pos, radii

    pos_fixed = pos.copy()
    x0 = radii.copy()
    n = N

    ii, jj = np.triu_indices(n, k=1)
    dx = pos_fixed[ii, 0] - pos_fixed[jj, 0]
    dy = pos_fixed[ii, 1] - pos_fixed[jj, 1]
    dist_sq = dx * dx + dy * dy
    dist_vals = np.sqrt(dist_sq)

    def obj(x):
        return -np.sum(x)

    def constraint_boundary(x):
        c1 = pos_fixed[:, 0] - x
        c2 = 1.0 - pos_fixed[:, 0] - x
        c3 = pos_fixed[:, 1] - x
        c4 = 1.0 - pos_fixed[:, 1] - x
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(x):
        return dist_sq - (x[ii] + x[jj])**2

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    bounds = [(1e-8, 0.5)] * n

    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
        if res.success or res.fun < obj(x0):
            new_radii = np.maximum(res.x, 1e-8)
            # Verify validity
            if _is_valid(pos_fixed, new_radii, tol=1e-9):
                return pos_fixed, new_radii
    except Exception:
        pass

    return pos, radii


def _final_slsqp_polish(pos, radii, deadline):
    """
    Final high-precision SLSQP polish with hard constraints.
    """
    if time.time() > deadline - 2:
        return pos, radii

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-8, 0.5))

    ii, jj = np.triu_indices(N, k=1)

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
        dx = p[ii, 0] - p[jj, 0]
        dy = p[ii, 1] - p[jj, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[ii] + r[jj]
        return dist_sq - min_dist * min_dist

    def objective(x):
        return -np.sum(x[2*N:])

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'disp': False})
        if res.success or res.fun < objective(x0):
            pos_new = res.x[:2*N].reshape(N, 2)
            rad_new = np.maximum(res.x[2*N:], 1e-8)
            pos_pp, rad_pp = _post_process(pos_new, rad_new)
            if _is_valid(pos_pp, rad_pp, tol=1e-9):
                return pos_pp, rad_pp
    except Exception:
        pass

    return pos, radii


def _best_fit_init(seed=0):
    """
    Greedy Best-Fit initialization.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

        for _ in range(20000):
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
                min_dist = min(x, 1-x, y, 1-y)
                for j in range(i):
                    d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
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


def _local_search(pos, radii, deadline, max_iter=5000, step_scale=1.0):
    """
    Simple local search with random perturbations to escape local minima.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_pos = pos.copy()
    current_radii = radii.copy()

    rng = np.random.RandomState(42)
    temp = 0.01
    cooling = 0.999

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        # Random perturbation
        new_pos = current_pos.copy()
        new_radii = current_radii.copy()

        # Perturb a random subset of circles
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, n_perturb, replace=False)
        for idx in indices:
            new_pos[idx, 0] += rng.uniform(-0.02, 0.02) * step_scale
            new_pos[idx, 1] += rng.uniform(-0.02, 0.02) * step_scale
            new_radii[idx] *= (1 + rng.uniform(-0.05, 0.05) * step_scale)
            new_radii[idx] = max(new_radii[idx], 1e-6)

        # Post-process to ensure validity
        new_pos_pp, new_radii_pp = _post_process(new_pos, new_radii)
        if _is_valid(new_pos_pp, new_radii_pp):
            new_score = np.sum(new_radii_pp)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos_pp.copy()
                best_radii = new_radii_pp.copy()
            # Accept with probability if worse
            delta = new_score - best_score
            if delta < 0 and rng.uniform() < np.exp(delta / max(temp, 1e-10)):
                current_pos = new_pos_pp.copy()
                current_radii = new_radii_pp.copy()
            elif new_score >= best_score:
                current_pos = new_pos_pp.copy()
                current_radii = new_radii_pp.copy()
        else:
            # Reject invalid
            pass

        temp *= cooling

    return best_pos, best_radii


def _optimize(pos, radii, deadline, seed=0):
    """
    Optimize the configuration using L-BFGS-B with penalty, then local search.
    """
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-8, 0.5))

    best_x = x0.copy()
    best_score = np.sum(radii)

    # Stage 1: Moderate penalty with soft-max
    p1 = 100.0
    beta1 = 50.0
    try:
        res1 = minimize(_penalty_function, x0, method='L-BFGS-B', bounds=bounds,
                        args=(p1, beta1),
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
        if res1.fun < _penalty_function(x0, p1, beta1):
            x_curr = res1.x
            pos_curr = x_curr[:2*N].reshape(N, 2)
            rad_curr = np.maximum(x_curr[2*N:], 1e-8)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_pp.flatten(), rad_pp])
    except Exception:
        pass

    # Stage 2: High penalty with softer max
    if time.time() < deadline - 10:
        p2 = 1000.0
        beta2 = 100.0
        try:
            res2 = minimize(_penalty_function, best_x, method='L-BFGS-B', bounds=bounds,
                            args=(p2, beta2),
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res2.fun < _penalty_function(best_x, p2, beta2):
                x_curr = res2.x
                pos_curr = x_curr[:2*N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2*N:], 1e-8)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints
    if time.time() < deadline - 8:
        try:
            ii, jj = np.triu_indices(N, k=1)

            def constraint_boundary(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                c1 = pos[:, 0] - radii
                c2 = 1.0 - pos[:, 0] - radii
                c3 = pos[:, 1] - radii
                c4 = 1.0 - pos[:, 1] - radii
                return np.concatenate([c1, c2, c3, c4])

            def constraint_overlap(x):
                pos = x[:2 * N].reshape(N, 2)
                radii = x[2 * N:]
                dx = pos[ii, 0] - pos[jj, 0]
                dy = pos[ii, 1] - pos[jj, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[ii] + radii[jj]
                return dist_sq - min_dist * min_dist

            def objective(x):
                return -np.sum(x[2*N:])

            constraints = [
                {'type': 'ineq', 'fun': constraint_boundary},
                {'type': 'ineq', 'fun': constraint_overlap}
            ]

            res3 = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})

            if res3.success or (res3.fun < objective(best_x)):
                pos_curr = res3.x[:2*N].reshape(N, 2)
                rad_curr = np.maximum(res3.x[2*N:], 1e-8)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 4: Local search to polish
    if time.time() < deadline - 6:
        pos_curr = best_x[:2*N].reshape(N, 2)
        rad_curr = best_x[2*N:]
        try:
            pos_ls, rad_ls = _local_search(pos_curr, rad_curr, deadline, max_iter=2000)
            if _is_valid(pos_ls, rad_ls):
                score = np.sum(rad_ls)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_ls.flatten(), rad_ls])
        except Exception:
            pass

    # Stage 5: Radius growth pass (fix positions, maximize radii)
    if time.time() < deadline - 4:
        pos_curr = best_x[:2*N].reshape(N, 2)
        rad_curr = best_x[2*N:]
        try:
            pos_rg, rad_rg = _radius_growth_pass(pos_curr, rad_curr, deadline)
            if _is_valid(pos_rg, rad_rg):
                score = np.sum(rad_rg)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_rg.flatten(), rad_rg])
        except Exception:
            pass

    # Stage 6: Final SLSQP polish
    if time.time() < deadline - 2:
        pos_curr = best_x[:2*N].reshape(N, 2)
        rad_curr = best_x[2*N:]
        try:
            pos_fp, rad_fp = _final_slsqp_polish(pos_curr, rad_curr, deadline)
            if _is_valid(pos_fp, rad_fp):
                score = np.sum(rad_fp)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_fp.flatten(), rad_fp])
        except Exception:
            pass

    pos_out = best_x[:2*N].reshape(N, 2)
    rad_out = best_x[2*N:]
    return pos_out, rad_out


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pos = None
    best_rad = None
    best_score = 0.0

    # Generate initializations
    inits = []

    # 1. Best Fit
    pos_bf, rad_bf = _best_fit_init(seed=1)
    inits.append((pos_bf, rad_bf))

    # 2. Best Fit with different seed
    pos_bf2, rad_bf2 = _best_fit_init(seed=2)
    inits.append((pos_bf2, rad_bf2))

    # 3. Hexagonal
    pos_hex, rad_hex = _hex_init()
    inits.append((pos_hex, rad_hex))

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

    best_pos, best_rad = _post_process(best_pos, best_rad)

    result = np.column_stack([best_pos, best_rad])
    return result

# EVOLVE-BLOCK-END
