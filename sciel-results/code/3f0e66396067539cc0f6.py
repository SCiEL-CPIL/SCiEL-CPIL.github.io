"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# Precompute pair indices
_II, _JJ = np.triu_indices(N, k=1)
_NPAIRS = len(_II)


def _soft_max(v, beta):
    """Smooth approximation of max(0, v) via softplus."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 20.0 / beta, v, np.log1p(np.exp(beta * v)) / beta)


def _vectorized_check(pos, radii, tol=1e-10):
    """
    Fully vectorized validity check. Returns (is_valid, min_margin).
    min_margin is the minimum of all constraint margins (positive = valid).
    """
    # Boundary margins
    m_left = pos[:, 0] - radii
    m_right = 1.0 - pos[:, 0] - radii
    m_bottom = pos[:, 1] - radii
    m_top = 1.0 - pos[:, 1] - radii
    boundary_min = min(m_left.min(), m_right.min(), m_bottom.min(), m_top.min())

    # Overlap margins: dist - (ri + rj)
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap_margin = dist - (radii[_II] + radii[_JJ])
    overlap_min = overlap_margin.min()

    overall_min = min(boundary_min, overlap_min)
    return overall_min > -tol, overall_min


def _penalty_function(x, penalty_weight=200.0, beta=80.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    Uses softplus for smooth max(0, v).
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-7)

    # Boundary violations
    bv_left = radii - pos[:, 0]
    bv_right = pos[:, 0] + radii - 1.0
    bv_bottom = radii - pos[:, 1]
    bv_top = pos[:, 1] + radii - 1.0

    boundary_pen = (np.sum(_soft_max(bv_left, beta) ** 2) +
                    np.sum(_soft_max(bv_right, beta) ** 2) +
                    np.sum(_soft_max(bv_bottom, beta) ** 2) +
                    np.sum(_soft_max(bv_top, beta) ** 2))

    # Overlap violations in distance space
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[_II] + radii[_JJ]
    viol = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol, beta) ** 2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)


def _post_process(pos, radii):
    """
    Ensure strict validity by minimally adjusting radii/positions.
    Vectorized overlap check for speed.
    """
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        changed = False

        # Boundary fix
        max_r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                                 np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over = radii > max_r_bound
        if np.any(over):
            radii[over] = max_r_bound[over] * (1.0 - 1e-10)
            changed = True

        # Overlap fix (vectorized detection, then fix)
        dx = pos[_II, 0] - pos[_JJ, 0]
        dy = pos[_II, 1] - pos[_JJ, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        total_r = radii[_II] + radii[_JJ]
        overlap = dist < total_r
        if np.any(overlap):
            for idx in np.where(overlap)[0]:
                i, j = _II[idx], _JJ[idx]
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                tr = radii[i] + radii[j]
                if tr > 1e-15:
                    scale = d / tr * (1.0 - 1e-10)
                    radii[i] *= scale
                    radii[j] *= scale
            changed = True

        if not changed:
            break

    # Final safety shrink
    valid, margin = _vectorized_check(pos, radii, tol=1e-10)
    if not valid:
        for _ in range(50):
            radii *= (1.0 - 1e-7)
            valid, margin = _vectorized_check(pos, radii, tol=1e-10)
            if valid:
                break

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

        for _ in range(30000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)

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

        if found and best_r > 0.005:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.97
        else:
            # Fallback: find any valid spot
            for _ in range(2000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                min_dist = min(x, 1 - x, y, 1 - y)
                for j in range(i):
                    d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                    min_dist = min(min_dist, d - radii[j])
                if min_dist > 0.003:
                    pos[i] = [x, y]
                    radii[i] = min_dist * 0.93
                    break
            else:
                pos[i] = [0.5, 0.5]
                radii[i] = 0.001

    return pos, radii


def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    r = 0.115
    dx = r * np.sqrt(3)
    dy = r * 1.5

    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r

        for col in range(6):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break

    rng = np.random.RandomState(456)
    while k < N:
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(8000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)
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
        if best_r > 0.005:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.97
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        k += 1

    return pos, radii


def _grid_init():
    """Grid-based initialization with varying radii."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    n = 5
    r = 0.095
    k = 0
    for i in range(n):
        for j in range(n):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / n, (j + 0.5) / n]
            radii[k] = r
            k += 1
    return pos, radii


def _force_directed_relax(pos, radii, deadline, max_iter=2000, damping=0.85):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Uses simple repulsive forces for overlaps and spring forces for boundaries.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Precompute pairs for efficiency
    pairs = list(zip(_II, _JJ))
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        forces = np.zeros_like(pos)
        
        # 1. Boundary forces (push inwards if too close to wall)
        mask = pos[:, 0] < radii
        forces[mask, 0] += (radii[mask] - pos[mask, 0]) * 2.0
        mask = pos[:, 0] > 1.0 - radii
        forces[mask, 0] -= (pos[mask, 0] - (1.0 - radii[mask])) * 2.0
        mask = pos[:, 1] < radii
        forces[mask, 1] += (radii[mask] - pos[mask, 1]) * 2.0
        mask = pos[:, 1] > 1.0 - radii
        forces[mask, 1] -= (pos[mask, 1] - (1.0 - radii[mask])) * 2.0
        
        # 2. Overlap forces (repel if dist < r_i + r_j)
        for i, j in pairs:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            
            if dist_sq < min_dist * min_dist and dist_sq > 1e-12:
                dist = np.sqrt(dist_sq)
                overlap = min_dist - dist
                fx = dx / dist
                fy = dy / dist
                forces[i, 0] += fx * overlap * 2.0
                forces[i, 1] += fy * overlap * 2.0
                forces[j, 0] -= fx * overlap * 2.0
                forces[j, 1] -= fy * overlap * 2.0
            elif dist_sq <= 1e-12:
                forces[i, 0] += np.random.uniform(-0.01, 0.01)
                forces[i, 1] += np.random.uniform(-0.01, 0.01)
                forces[j, 0] += np.random.uniform(-0.01, 0.01)
                forces[j, 1] += np.random.uniform(-0.01, 0.01)

        pos += forces * 0.05 * damping
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        max_r_allowed = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]), 
                                   np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over = radii > max_r_allowed * 1.05
        if np.any(over):
            radii[over] = max_r_allowed[over] * 0.95

    return pos, radii


def _local_search(pos, radii, deadline, max_iter=10000, step_scale=1.0):
    """
    Fast local search with vectorized validation and simulated annealing.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_pos = pos.copy()
    current_radii = radii.copy()

    rng = np.random.RandomState(777)
    temp = 0.008
    cooling = 0.9995

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        new_pos = current_pos.copy()
        new_radii = current_radii.copy()

        n_perturb = rng.randint(1, 5)
        indices = rng.choice(N, n_perturb, replace=False)
        for idx in indices:
            new_pos[idx, 0] += rng.uniform(-0.015, 0.015) * step_scale
            new_pos[idx, 1] += rng.uniform(-0.015, 0.015) * step_scale
            new_radii[idx] *= (1 + rng.uniform(-0.05, 0.05) * step_scale)
            new_radii[idx] = max(new_radii[idx], 1e-7)

        valid, margin = _vectorized_check(new_pos, new_radii, tol=-1e-9)
        if valid:
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            delta = new_score - best_score
            if delta < 0 and rng.uniform() < np.exp(delta / max(temp, 1e-12)):
                current_pos = new_pos.copy()
                current_radii = new_radii.copy()
            elif new_score >= best_score:
                current_pos = new_pos.copy()
                current_radii = new_radii.copy()

        temp *= cooling

    return best_pos, best_radii


def _slsqp_pass(pos, radii, deadline, maxiter=500):
    """SLSQP with hard constraints."""
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
                       options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16, 'disp': False})
        if res.success or (res.fun < objective(x0)):
            pos_curr = res.x[:2 * N].reshape(N, 2)
            rad_curr = np.maximum(res.x[2 * N:], 1e-7)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            valid, _ = _vectorized_check(pos_pp, rad_pp)
            if valid:
                return pos_pp, rad_pp
    except Exception:
        pass

    return pos, radii


def _final_refinement(pos, radii, deadline):
    """
    High-precision final refinement: multiple SLSQP passes with tighter tolerances
    and a final local search.
    """
    best_pos = pos.copy()
    best_rad = radii.copy()
    best_score = np.sum(radii)

    # SLSQP pass 1
    if time.time() < deadline - 15:
        try:
            p1, r1 = _slsqp_pass(best_pos, best_rad, deadline - 15, maxiter=400)
            valid, _ = _vectorized_check(p1, r1)
            if valid:
                s = np.sum(r1)
                if s > best_score:
                    best_score = s
                    best_pos = p1.copy()
                    best_rad = r1.copy()
        except Exception:
            pass

    # SLSQP pass 2 (from current best)
    if time.time() < deadline - 10:
        try:
            p2, r2 = _slsqp_pass(best_pos, best_rad, deadline - 10, maxiter=300)
            valid, _ = _vectorized_check(p2, r2)
            if valid:
                s = np.sum(r2)
                if s > best_score:
                    best_score = s
                    best_pos = p2.copy()
                    best_rad = r2.copy()
        except Exception:
            pass

    # Final local search
    if time.time() < deadline - 5:
        try:
            p3, r3 = _local_search(best_pos, best_rad, deadline - 5, max_iter=5000)
            valid, _ = _vectorized_check(p3, r3)
            if valid:
                s = np.sum(r3)
                if s > best_score:
                    best_score = s
                    best_pos = p3.copy()
                    best_rad = r3.copy()
        except Exception:
            pass

    return best_pos, best_rad


def _optimize(pos, radii, deadline, seed=0):
    """
    Optimize the configuration using multi-stage approach.
    """
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    best_x = x0.copy()
    best_score = np.sum(radii)

    # Stage 0: Force-directed relaxation
    if time.time() < deadline - 30:
        pos_fd, rad_fd = _force_directed_relax(pos, radii, deadline - 30)
        pos_pp_fd, rad_pp_fd = _post_process(pos_fd, rad_fd)
        valid_fd, _ = _vectorized_check(pos_pp_fd, rad_pp_fd)
        if valid_fd:
            score_fd = np.sum(rad_pp_fd)
            if score_fd > best_score:
                best_score = score_fd
                best_x = np.concatenate([pos_pp_fd.flatten(), rad_pp_fd])
            pos = pos_pp_fd
            radii = rad_pp_fd
            x0 = np.concatenate([pos.flatten(), radii])

    # Stage 1: L-BFGS-B with moderate penalty
    if time.time() < deadline - 20:
        p1 = 150.0
        beta1 = 60.0
        try:
            res1 = minimize(_penalty_function, x0, method='L-BFGS-B', bounds=bounds,
                            args=(p1, beta1),
                            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
            if res1.fun < _penalty_function(x0, p1, beta1):
                x_curr = res1.x
                pos_curr = x_curr[:2 * N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2 * N:], 1e-7)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                valid, _ = _vectorized_check(pos_pp, rad_pp)
                if valid:
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 2: L-BFGS-B with higher penalty
    if time.time() < deadline - 15:
        p2 = 500.0
        beta2 = 100.0
        try:
            res2 = minimize(_penalty_function, best_x, method='L-BFGS-B', bounds=bounds,
                            args=(p2, beta2),
                            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
            if res2.fun < _penalty_function(best_x, p2, beta2):
                x_curr = res2.x
                pos_curr = x_curr[:2 * N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2 * N:], 1e-7)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                valid, _ = _vectorized_check(pos_pp, rad_pp)
                if valid:
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints
    if time.time() < deadline - 10:
        try:
            pos_curr = best_x[:2 * N].reshape(N, 2)
            rad_curr = best_x[2 * N:]
            pos_sls, rad_sls = _slsqp_pass(pos_curr, rad_curr, deadline - 10, maxiter=400)
            valid, _ = _vectorized_check(pos_sls, rad_sls)
            if valid:
                score = np.sum(rad_sls)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_sls.flatten(), rad_sls])
        except Exception:
            pass

    # Stage 4: Fast local search
    if time.time() < deadline - 5:
        pos_curr = best_x[:2 * N].reshape(N, 2)
        rad_curr = best_x[2 * N:]
        try:
            pos_ls, rad_ls = _local_search(pos_curr, rad_curr, deadline, max_iter=8000)
            valid, _ = _vectorized_check(pos_ls, rad_ls)
            if valid:
                score = np.sum(rad_ls)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_ls.flatten(), rad_ls])
        except Exception:
            pass

    pos_out = best_x[:2 * N].reshape(N, 2)
    rad_out = best_x[2 * N:]
    return pos_out, rad_out


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin

    best_pos = None
    best_rad = None
    best_score = 0.0

    # Generate diverse initializations
    inits = []

    # 1. Best Fit seed 1
    pos_bf, rad_bf = _best_fit_init(seed=1)
    inits.append((pos_bf, rad_bf))

    # 2. Best Fit seed 42
    pos_bf2, rad_bf2 = _best_fit_init(seed=42)
    inits.append((pos_bf2, rad_bf2))

    # 3. Hexagonal
    pos_hex, rad_hex = _hex_init()
    inits.append((pos_hex, rad_hex))

    # 4. Grid
    pos_grid, rad_grid = _grid_init()
    inits.append((pos_grid, rad_grid))

    # Optimize each initialization
    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 35:
            break

        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        valid, _ = _vectorized_check(pos_pp, rad_pp)
        if valid:
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_rad = rad_pp.copy()

        try:
            pos_opt, rad_opt = _optimize(pos_pp, rad_pp, deadline, seed=i)
            pos_pp_opt, rad_pp_opt = _post_process(pos_opt, rad_opt)
            valid, _ = _vectorized_check(pos_pp_opt, rad_pp_opt)
            if valid:
                score = np.sum(rad_pp_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp_opt.copy()
                    best_rad = rad_pp_opt.copy()
        except Exception:
            pass

    # Final high-precision refinement on the best solution
    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_ref, rad_ref = _final_refinement(best_pos, best_rad, deadline)
            pos_ref, rad_ref = _post_process(pos_ref, rad_ref)
            valid, _ = _vectorized_check(pos_ref, rad_ref)
            if valid:
                score = np.sum(rad_ref)
                if score > best_score:
                    best_score = score
                    best_pos = pos_ref.copy()
                    best_rad = rad_ref.copy()
        except Exception:
            pass

    # Final local search polish with Jittering to escape local minima
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_ls, rad_ls = _local_search(best_pos, best_rad, deadline - 10, max_iter=3000)
            valid, _ = _vectorized_check(pos_ls, rad_ls)
            if valid:
                score = np.sum(rad_ls)
                if score > best_score:
                    best_score = score
                    best_pos = pos_ls.copy()
                    best_rad = rad_ls.copy()

            # Jittered restarts
            while time.time() < deadline - 10:
                rng = np.random.RandomState(int(time.time() * 1000) % 10000)
                pos_jit = best_pos + rng.normal(0, 0.005, (N, 2))
                rad_jit = best_rad * (1 + rng.normal(0, 0.01, N))
                rad_jit = np.maximum(rad_jit, 1e-7)

                pos_pol, rad_pol = _slsqp_pass(pos_jit, rad_jit, deadline - 5, maxiter=200)
                pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
                valid_jit, _ = _vectorized_check(pos_pol, rad_pol)
                if valid_jit:
                    score_jit = np.sum(rad_pol)
                    if score_jit > best_score:
                        best_score = score_jit
                        best_pos = pos_pol.copy()
                        best_rad = rad_pol.copy()
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
