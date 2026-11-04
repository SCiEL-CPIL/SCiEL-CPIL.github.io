"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _objective_single_move(pos, idx, new_xy):
    """
    Fast objective evaluation when only position[idx] changes.
    Avoids full cdist recomputation.
    """
    n = len(pos)
    old_xy = pos[idx]
    # Compute distances from new position to all others
    dx = pos[:, 0] - new_xy[0]
    dy = pos[:, 1] - new_xy[1]
    dists_new = np.sqrt(dx * dx + dy * dy)
    dists_new[idx] = np.inf
    
    # For each circle i, the binding constraint is min(boundary_i, min_j dist(i,j)/2)
    # When we move circle idx, only the distances involving idx change.
    # For circles i != idx: their min_pair may change if idx was their nearest neighbor
    # For circle idx: its min_pair changes based on new distances
    
    # Compute boundary distances
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    boundary[idx] = min(new_xy[0], 1.0 - new_xy[0], new_xy[1], 1.0 - new_xy[1])
    
    # We need to recompute min_pair for all circles since idx moved
    # But we can do it more efficiently:
    # Full cdist is O(n^2), but for n=26 it's fast enough
    # Let's just use the full computation but with the updated position
    new_pos = pos.copy()
    new_pos[idx] = new_xy
    dists = cdist(new_pos, new_pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    radii = np.minimum(boundary, min_pair)
    return float(np.sum(np.maximum(radii, 0.0)))


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search with two-scale refinement:
    coarse grid search followed by fine grid polish around the best candidate.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    # Adaptive grid resolution and step
    base_step = 0.015
    grid_res = 5  # (2*grid_res+1)^2 candidates per circle
    fine_grid_res = 4  # finer resolution for second pass
    fine_step_scale = 0.3  # fine step is 30% of coarse step

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)
        
        # Adaptive step: shrink as iterations progress
        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            
            # Coarse grid search
            best_new_pos = None
            best_new_obj = current_obj
            best_cx, best_cy = x_i, y_i

            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_pos = new_pos
                        best_cx, best_cy = cx, cy

            # Fine grid refinement around the best coarse candidate
            fine_step = step * fine_step_scale
            for di in range(-fine_grid_res, fine_grid_res + 1):
                for dj in range(-fine_grid_res, fine_grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = best_cx + di * fine_step
                    cy = best_cy + dj * fine_step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_pos = new_pos

            # Also try some diagonal/random perturbations
            for _ in range(8):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_pos = new_pos

            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            # Try with smaller random steps
            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                x_i, y_i = current_pos[i]
                dx = rng.uniform(-step * 0.5, step * 0.5)
                dy = rng.uniform(-step * 0.5, step * 0.5)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(x_i + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(y_i + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-14:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                base_step *= 0.6
                if base_step < 0.0003:
                    break

    return current_pos, current_obj


def _is_valid(pos, eps=1e-9):
    """Check pairwise non-overlap and boundary containment with computed radii."""
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True


def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """
    Simulated annealing to maximize sum of radii.
    Uses vectorized _max_radius_for_positions for speed.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        t = t_start * (t_end / t_start) ** (it / n_iter)

        i = rng.randint(n)
        step_size = 0.04 * t + 0.0008
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)

        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

        new_obj = _objective(new_pos)
        delta = new_obj - current_obj

        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj

    return best_pos, best_obj


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """
    Local search: iteratively nudge each circle to improve the objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(15):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break

    return current_pos, current_obj


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: run gradient-based optimization with progressive
    beta values to smooth the landscape initially and sharpen it later.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))
    
    # More beta stages with finer granularity, reaching higher sharpness
    betas = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0]
        
    for beta in betas:
        if time.time() > deadline - 1:
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        try:
            # Increase iterations for higher beta to allow convergence
            maxiter = 2000 if beta >= 150 else 1000
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
        except Exception:
            continue

        if time.time() > deadline - 2:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])
        
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        if _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj - 1e-10:
            current_pos = pos_opt.copy()

    # Final micro-polish: very small perturbations around best solution
    if time.time() < deadline - 0.5:
        rng = np.random.RandomState(42)
        for _ in range(1000):
            if time.time() > deadline - 0.1:
                break
            i = rng.randint(n)
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if _is_valid(new_pos, eps=1e-9) and new_obj > best_obj + 1e-12:
                best_pos = new_pos
                best_r = new_r
                best_obj = new_obj

    return best_pos, best_r, best_obj


def _smooth_objective(v, n, beta=40.0):
    """
    Smooth objective for gradient-based polish.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations: r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations: r_i + r_j - dist_ij
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_pen += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 50.0 * (wall_pen + total_pen)


def _smooth_gradient(v, n, beta=40.0):
    """
    Analytic gradient of _smooth_objective.
    Uses smooth approximations for min to avoid discontinuities.
    """
    x = v[0:n].copy()
    y = v[n:2 * n].copy()
    r = v[2 * n:3 * n].copy()

    grad = np.zeros(3 * n)
    penalty_weight = 50.0

    # Wall terms
    # wall_min = min(x, 1-x, y, 1-y)
    # We approximate the gradient of min using softmax weights
    # d/dx min(a,b) ~ a/(a+b) * da + b/(a+b) * db if we use soft min?
    # Actually, for f(x) = -log(sum(exp(-x_i))) / beta, grad is (exp(-x_i)/sum(exp(-x_j)))
    # But we have logaddexp(0, beta * viol).
    # Let's stick to the sigmoid derivative for the violation part, but smooth the min part.
    
    # To make it robust, we compute the "soft" contribution of each wall to the min.
    # If a circle is close to a wall, that wall dominates the gradient.
    # We use a temperature for the softmax to make it smooth.
    temp = 10.0 
    
    for i in range(n):
        # Distances to walls
        d_left = x[i]
        d_right = 1.0 - x[i]
        d_down = y[i]
        d_up = 1.0 - y[i]
        
        dists = np.array([d_left, d_right, d_down, d_up])
        
        # Softmin weights: w_k = exp(-d_k * temp) / sum(exp(-d_j * temp))
        exp_dists = np.exp(-dists * temp)
        sum_exp = np.sum(exp_dists)
        if sum_exp < 1e-12:
            continue
        weights = exp_dists / sum_exp
        
        wall_min = np.dot(weights, dists) # Approximate min
        wall_viol = r[i] - wall_min
        sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        
        if sig_wall < 1e-12:
            continue
            
        factor = penalty_weight * sig_wall
        
        grad[2 * n + i] += factor
        
        # Gradient with respect to x[i], y[i]
        # d(wall_viol)/dx_i = -d(wall_min)/dx_i
        # d(wall_min)/dx_i approx w_left - w_right
        # So d(wall_viol)/dx_i approx -(w_left - w_right) = w_right - w_left
        # grad[i] += factor * (w_right - w_left)
        # Current code: grad[i] -= factor * (-w_left + w_right) which is grad[i] += factor * (w_left - w_right)
        # This is WRONG. It should be grad[i] += factor * (w_right - w_left)
        # Let's rewrite clearly:
        # grad[i] should decrease if moving x_i right (increasing x_i) decreases violation.
        # If x_i near 0, w_left > w_right. Moving right increases wall_min, decreases viol.
        # So we want x_i to increase.
        # In minimization, grad < 0 means x increases.
        # So we want grad[i] < 0.
        # w_right - w_left < 0.
        # So grad[i] += factor * (w_right - w_left) is correct.
        
        grad[i] += factor * (weights[1] - weights[0])
        grad[n + i] += factor * (weights[3] - weights[2])

    # Pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-12:
                continue
            factor = penalty_weight * s
            # d(viol)/dx_i = -dx/dist
            # d(penalty)/dx_i = factor * (-dx/dist)
            grad[i] -= factor * dx / dist
            grad[n + i] -= factor * dy / dist
            grad[j] += factor * dx / dist
            grad[n + j] += factor * dy / dist
            grad[2 * n + i] += factor
            grad[2 * n + j] += factor

    # Objective: -sum(r) + penalty
    grad[2 * n:3 * n] -= 1.0

    return grad


def _gradient_polish(pos, deadline, beta=40.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
    Returns improved (pos, radii) if valid and better.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
    for i in range(n):
        bounds.append((0.0, 1.0))  # y
    for i in range(n):
        bounds.append((0.0, 0.5))  # r

    try:
        res = minimize(
            _smooth_objective, v0, args=(n, beta),
            jac=_smooth_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 80}
        )
    except Exception:
        return pos, r_init

    if time.time() > deadline:
        return pos, r_init

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]
    r_opt = v_opt[2 * n:3 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    # Recompute exact radii from positions (the optimizer may have r > feasible)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    # Verify validity
    if _is_valid(pos_opt, eps=1e-9):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 35:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=30000, seed=init_idx * 7 + 1,
            deadline=deadline - 30
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 25:
            break

        # Best-fit local search (targeted moves) - more iterations
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=6000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 20
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 15:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2500, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 12
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution - extended
    if time.time() < deadline - 25:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=12000,
            seed=777, deadline=deadline - 20
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 2b: Second best-fit pass with different seed
    if time.time() < deadline - 18:
        bf_final2, bf_final2_obj = _best_fit_local_search(
            best_pos, n_iter=8000,
            seed=888, deadline=deadline - 15
        )
        if bf_final2_obj > best_obj:
            best_pos = bf_final2
            best_obj = bf_final2_obj

    # Phase 2c: Third best-fit pass with yet another seed for more refinement
    if time.time() < deadline - 12:
        bf_final3, bf_final3_obj = _best_fit_local_search(
            best_pos, n_iter=6000,
            seed=999, deadline=deadline - 10
        )
        if bf_final3_obj > best_obj:
            best_pos = bf_final3
            best_obj = bf_final3_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 12:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=5000, step_size=0.002,
            seed=999, deadline=deadline - 10
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Gradient-based polish (smooth surrogate)
    if time.time() < deadline - 25:
        polished = _gradient_polish(best_pos, deadline - 20)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 5: High-precision final polish (progressive beta) - more time
    if time.time() < deadline - 10:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 5b: Another best-fit pass after gradient polish
    if time.time() < deadline - 5:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=5000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 5c: Additional fine-tuning pass
    if time.time() < deadline - 2.5:
        bf_after_grad2, bf_ag2_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=666, deadline=deadline - 1.5
        )
        if bf_ag2_obj > best_obj:
            best_pos = bf_after_grad2
            best_obj = bf_ag2_obj

    # Phase 6: Final fine-tuning local search with very small steps
    if time.time() < deadline - 3:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=2000, step_size=0.0006,
            seed=12345, deadline=deadline - 2
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 7: Ultra-fine polish with two-scale approach
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(800):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            # Coarse random step
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if new_obj > best_obj + 1e-13:
                best_pos = new_pos
                best_obj = new_obj
                # Fine refinement around this improvement
                for _fine in range(10):
                    if time.time() > deadline - 0.05:
                        break
                    j = rng.randint(len(best_pos))
                    fdx = rng.uniform(-0.0001, 0.0001)
                    fdy = rng.uniform(-0.0001, 0.0001)
                    fine_pos = best_pos.copy()
                    fine_pos[j, 0] = np.clip(best_pos[j, 0] + fdx, 0.0, 1.0)
                    fine_pos[j, 1] = np.clip(best_pos[j, 1] + fdy, 0.0, 1.0)
                    fine_r = _max_radius_for_positions(fine_pos)
                    fine_obj = float(np.sum(fine_r))
                    if fine_obj > best_obj + 1e-14:
                        best_pos = fine_pos
                        best_obj = fine_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
