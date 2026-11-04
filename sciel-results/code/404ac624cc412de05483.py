"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
_START = None
_DEADLINE = 140.0

# EVOLVE-BLOCK-START

def _elapsed():
    return time.time() - _START if _START is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


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


def _soft_objective_vec(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


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


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est
    
    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    rs_arr = np.array(rs)
    
    min_x = np.min(xs_arr - rs_arr)
    max_x = np.max(xs_arr + rs_arr)
    min_y = np.min(ys_arr - rs_arr)
    max_y = np.max(ys_arr + rs_arr)
    
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)
    
    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    rs_arr = rs_arr * scale
    
    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2
    
    pos = np.column_stack([xs_arr, ys_arr])
    return pos


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


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search: for each circle, search a fine grid
    around its current position to find the best new location.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    base_step = 0.015
    grid_res = 5

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)
        
        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            
            best_new_pos = None
            best_new_obj = current_obj

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


def _smooth_objective(v, n, beta=40.0):
    """
    Smooth objective for gradient-based polish.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    Vectorized pairwise computation.
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Vectorized pairwise
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    viol = r[i_idx] + r[j_idx] - dist
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol)) / beta

    return -np.sum(r) + 50.0 * (wall_pen + pair_pen)


def _grow_radii(pos, deadline, n_iter=500):
    """
    After positions are fixed, iteratively try to grow each circle's radius
    while maintaining feasibility. This directly exploits slack in the packing.
    Uses vectorized distance checks for speed.
    """
    n = len(pos)
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if time.time() > deadline:
            break
        improved = False
        it += 1
        for i in range(n):
            if time.time() > deadline:
                break
            # Try to grow radius i by a small amount
            for frac in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]:
                new_r = radii[i] * (1.0 + frac)
                # Check if this radius is feasible
                feasible = True
                # Boundary check
                if (new_r > pos[i, 0] - 1e-10 or new_r > 1.0 - pos[i, 0] + 1e-10 or
                        new_r > pos[i, 1] - 1e-10 or new_r > 1.0 - pos[i, 1] + 1e-10):
                    feasible = False
                    break
                # Pairwise check (vectorized)
                if np.any(new_r + radii > dists[i] + 1e-10):
                    feasible = False
                if feasible:
                    radii[i] = new_r
                    improved = True
                    break
    return radii


def _largest_empty_circle_refine(pos, radii, deadline, n_candidates=20):
    """
    For each circle, find the largest empty circle position nearby and
    potentially reposition it there if it increases the total radius sum.
    """
    n = len(pos)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_obj = float(np.sum(radii))

    for i in range(n):
        if time.time() > deadline:
            break

        # Sample candidate positions in a ring around current position
        x_i, y_i = pos[i]
        r_i = radii[i]

        # Candidate positions: grid around current + some random
        candidates = []
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for dist in [r_i * 0.3, r_i * 0.5, r_i * 0.8, r_i * 1.2, r_i * 1.5]:
                cx = x_i + dist * np.cos(angle)
                cy = y_i + dist * np.sin(angle)
                if 0 <= cx <= 1 and 0 <= cy <= 1:
                    candidates.append((cx, cy))

        # Also add the current position
        candidates.append((x_i, y_i))

        # Add random candidates
        rng = np.random.RandomState(i * 31 + 7)
        for _ in range(8):
            cx = rng.uniform(0, 1)
            cy = rng.uniform(0, 1)
            candidates.append((cx, cy))

        best_new_obj = best_obj
        best_new_pos = None

        for cx, cy in candidates:
            if time.time() > deadline:
                break
            new_pos = best_pos.copy()
            new_pos[i] = [cx, cy]
            new_radii = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_radii))
            if new_obj > best_new_obj + 1e-12:
                best_new_obj = new_obj
                best_new_pos = new_pos
                best_new_radii = new_radii

        if best_new_pos is not None:
            best_pos = best_new_pos
            best_radii = best_new_radii
            best_obj = best_new_obj

    return best_pos, best_radii, best_obj


def _smooth_gradient(v, n, beta=40.0):
    """
    Analytic gradient of _smooth_objective. Vectorized pairwise computation.
    """
    x = v[0:n].copy()
    y = v[n:2 * n].copy()
    r = v[2 * n:3 * n].copy()

    grad = np.zeros(3 * n)

    # Wall penalty gradients (vectorized)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    sig_wall = np.where(sig_wall > 1e-12, sig_wall, 0.0)
    factor_wall = 50.0 * beta * sig_wall

    # Determine which wall is closest for each circle
    dists_to_walls = np.column_stack([x, 1.0 - x, y, 1.0 - y])
    min_idx = np.argmin(dists_to_walls, axis=1)

    # d wall_min / dx_i
    d_wall_dx = np.where(min_idx == 0, -1.0, 0.0) + np.where(min_idx == 1, 1.0, 0.0)
    d_wall_dy = np.where(min_idx == 2, -1.0, 0.0) + np.where(min_idx == 3, 1.0, 0.0)

    grad[0:n] += factor_wall * d_wall_dx
    grad[n:2 * n] += factor_wall * d_wall_dy
    grad[2 * n:3 * n] += factor_wall

    # Pairwise penalty gradients (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq)
    safe_dist = np.where(dist < 1e-12, 1.0, dist)
    viol = r[i_idx] + r[j_idx] - dist
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    sig = np.where(sig > 1e-12, sig, 0.0)
    factor = 50.0 * beta * sig

    ux = dx / safe_dist
    uy = dy / safe_dist

    np.add.at(grad, i_idx, -factor * ux)
    np.add.at(grad, n + i_idx, -factor * uy)
    np.add.at(grad, j_idx, factor * ux)
    np.add.at(grad, n + j_idx, factor * uy)
    np.add.at(grad, 2 * n + i_idx, factor)
    np.add.at(grad, 2 * n + j_idx, factor)

    # Objective: -sum(r)
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
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(
            _smooth_objective, v0, args=(n, beta),
            jac=_smooth_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]
    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-9):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: run gradient-based optimization with progressive
    beta values to smooth the landscape initially and sharpen it later.
    Includes radius growth and empty-circle refinement.
    Uses vectorized smooth objective/gradient for speed.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))
    
    betas = [15.0, 30.0, 50.0, 80.0, 120.0, 180.0, 260.0, 380.0]
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    for beta in betas:
        if time.time() > deadline - 1.5:
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        try:
            maxiter = 1200 if beta >= 100 else 700
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 120}
            )
        except Exception:
            continue

        if time.time() > deadline - 1.5:
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

    # Radius growth step
    if time.time() < deadline - 1.0:
        grown_r = _grow_radii(best_pos, deadline - 0.5)
        grown_obj = float(np.sum(grown_r))
        if grown_obj > best_obj + 1e-12:
            best_r = grown_r
            best_obj = grown_obj

    # Largest empty circle refinement
    if time.time() < deadline - 1.5:
        lec_pos, lec_r, lec_obj = _largest_empty_circle_refine(
            best_pos, best_r, deadline - 0.5
        )
        if lec_obj > best_obj + 1e-12:
            best_pos = lec_pos
            best_r = lec_r
            best_obj = lec_obj

    # Final micro-adjustment
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(42)
        for _ in range(1200):
            if time.time() > deadline - 0.05:
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


def _vectorized_gradient_polish(pos, deadline):
    """
    Uses the vectorized soft objective from the inspiration.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    x0 = np.zeros(3 * n)
    x0[0::3] = pos[:, 0]
    x0[1::3] = pos[:, 1]
    x0[2::3] = r_init

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    try:
        res = minimize(
            _soft_objective_vec, x0, args=(n, 1000.0),
            jac=lambda x, n, mu: _soft_objective_vec(x, n, mu)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    x_opt = res.x
    pos_opt = np.column_stack([x_opt[0::3], x_opt[1::3]])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-9):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 140.0
    deadline = _START + 140.0

    # Diverse initializations
    initializations = [_initial_grid(), _hex_packing_init()]
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

        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=6000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 20
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

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

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 25:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=12000,
            seed=777, deadline=deadline - 20
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    if time.time() < deadline - 18:
        bf_final2, bf_final2_obj = _best_fit_local_search(
            best_pos, n_iter=8000,
            seed=888, deadline=deadline - 15
        )
        if bf_final2_obj > best_obj:
            best_pos = bf_final2
            best_obj = bf_final2_obj

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
    if time.time() < deadline - 30:
        polished = _gradient_polish(best_pos, deadline - 28)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 4b: Vectorized gradient polish
    if time.time() < deadline - 20:
        v_polished = _vectorized_gradient_polish(best_pos, deadline - 18)
        if len(v_polished) == 3:
            v_p, v_r, v_obj = v_polished
            if v_obj > best_obj:
                best_pos = v_p
                best_obj = v_obj

    # Phase 5: High-precision final polish (progressive beta)
    if time.time() < deadline - 18:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 8)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 5b: Another best-fit pass after gradient polish
    if time.time() < deadline - 8:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=5000,
            seed=555, deadline=deadline - 6
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 5c: Radius growth after best-fit
    if time.time() < deadline - 5:
        grown_r = _grow_radii(best_pos, deadline - 4)
        grown_obj = float(np.sum(grown_r))
        if grown_obj > best_obj + 1e-12:
            best_obj = grown_obj

    # Phase 5d: Largest empty circle refinement
    if time.time() < deadline - 5:
        lec_pos, lec_r, lec_obj = _largest_empty_circle_refine(
            best_pos, _max_radius_for_positions(best_pos), deadline - 4
        )
        if lec_obj > best_obj + 1e-12:
            best_pos = lec_pos
            best_obj = lec_obj

    # Phase 5e: Second high-precision polish after local improvements
    if time.time() < deadline - 10:
        hp2_pos, hp2_r, hp2_obj = _high_precision_polish(best_pos, deadline - 5)
        if hp2_obj > best_obj:
            best_pos = hp2_pos
            best_obj = hp2_obj

    # Phase 6: Final fine-tuning local search with very small steps
    if time.time() < deadline - 5:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=2000, step_size=0.0006,
            seed=12345, deadline=deadline - 4
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 6b: Grow radii after fine-tuning
    if time.time() < deadline - 3:
        grown_r2 = _grow_radii(best_pos, deadline - 2)
        grown_obj2 = float(np.sum(grown_r2))
        if grown_obj2 > best_obj + 1e-12:
            best_obj = grown_obj2

    # Phase 7: Ultra-fine polish
    if time.time() < deadline - 1.0:
        rng = np.random.RandomState(99999)
        for _ in range(800):
            if time.time() > deadline - 0.1:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0002, 0.0002)
            dy = rng.uniform(-0.0002, 0.0002)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if new_obj > best_obj + 1e-13:
                best_pos = new_pos
                best_obj = new_obj

    # Phase 7b: Final radius growth
    if time.time() < deadline - 0.1:
        final_grown = _grow_radii(best_pos, deadline)
        final_grown_obj = float(np.sum(final_grown))
        if final_grown_obj > best_obj + 1e-12:
            best_obj = final_grown_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
