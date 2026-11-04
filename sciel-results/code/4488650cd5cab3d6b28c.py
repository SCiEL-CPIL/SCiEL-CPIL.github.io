"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
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

def _project_feasible(pos, max_iter=100):
    """
    Iteratively adjust positions and radii to ensure strict feasibility.
    If circles overlap, shrink radii and push positions apart slightly.
    If circles hit boundary, shrink radius and push inwards.
    """
    pos = pos.copy()
    for _ in range(max_iter):
        radii = _max_radius_for_positions(pos)
        
        # Check for violations
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        
        valid = True
        for i in range(N):
            # Boundary check
            min_boundary = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > min_boundary + 1e-12:
                # Shrink radius to boundary
                radii[i] = min_boundary * 0.999
                valid = False
            
            # Pairwise check
            for j in range(i+1, N):
                dist = dists[i, j]
                if radii[i] + radii[j] > dist + 1e-12:
                    # Overlap exists. Shrink both radii slightly and push apart
                    # Simple push: move each by half the overlap in opposite directions
                    overlap = radii[i] + radii[j] - dist
                    if dist > 1e-12:
                        dx = (pos[i,0] - pos[j,0]) / dist
                        dy = (pos[i,1] - pos[j,1]) / dist
                        push = overlap / 2.0 + 1e-6
                        pos[i,0] += dx * push
                        pos[i,1] += dy * push
                        pos[j,0] -= dx * push
                        pos[j,1] -= dy * push
                    # Shrink radii to ensure separation
                    new_r_i = (dist - overlap) / 2.0 * 0.999
                    new_r_j = (dist - overlap) / 2.0 * 0.999
                    radii[i] = min(radii[i], new_r_i)
                    radii[j] = min(radii[j], new_r_j)
                    valid = False
        
        # Clip positions to [0,1]
        pos = np.clip(pos, 0.0, 1.0)
        
        if valid:
            break
            
    # Final re-computation of max radii to ensure they are the largest possible
    final_radii = _max_radius_for_positions(pos)
    return pos, final_radii

def _grow_radii(pos, max_iter=50):
    """
    Try to increase radii by solving for optimal radii given fixed positions?
    Actually, radii are determined by positions. This function is a placeholder 
    for a more complex local search that might adjust positions specifically 
    to allow radius growth, but since _objective already maximizes sum of radii 
    for given positions, this is less useful unless we change the variable space.
    Instead, we'll use a local search that specifically tries to increase the 
    minimum radius or sum of radii by small moves.
    """
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    rng = np.random.RandomState(42)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try small random moves
            for _ in range(10):
                dx = rng.uniform(-0.001, 0.001)
                dy = rng.uniform(-0.001, 0.001)
                new_pos = current_pos.copy()
                new_pos[i,0] = np.clip(current_pos[i,0] + dx, 0.0, 1.0)
                new_pos[i,1] = np.clip(current_pos[i,1] + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-12:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break
        if not improved:
            break
            
    return current_pos, current_obj

def _soft_objective(v, n, beta=40.0):
    """
    Vectorized smooth objective for gradient-based polish.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations: r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    # Smooth penalty: sum(logaddexp(0, beta * viol)) / beta
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations: r_i + r_j - dist_ij
    # Vectorized distance calculation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    viol = r_sum - dists
    # Only sum over upper triangle to avoid double counting
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    viol_masked = viol[mask]
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol_masked)) / beta

    return -np.sum(r) + 50.0 * (wall_pen + pair_pen)

def _soft_gradient(v, n, beta=40.0):
    """
    Vectorized analytic gradient of _soft_objective.
    """
    x = v[0:n].copy()
    y = v[n:2 * n].copy()
    r = v[2 * n:3 * n].copy()

    grad = np.zeros(3 * n)
    penalty_weight = 50.0

    # Wall terms
    # wall_min = min(x, 1-x, y, 1-y)
    # We use a soft-min approximation via softmax weights for the gradient.
    # d/dx min(a,b) ~ a/(a+b) * da + b/(a+b) * db
    # Here we have 4 walls.
    d_left = x
    d_right = 1.0 - x
    d_down = y
    d_up = 1.0 - y
    
    dists = np.stack([d_left, d_right, d_down, d_up], axis=1)
    # Softmin weights: w_k = exp(-d_k * temp) / sum(exp(-d_j * temp))
    temp = 10.0 
    exp_dists = np.exp(-dists * temp)
    sum_exp = np.sum(exp_dists, axis=1, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-12)
    weights = exp_dists / sum_exp # shape (n, 4)
    
    wall_min = np.sum(weights * dists, axis=1) # shape (n,)
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol)) # shape (n,)
    
    factor = penalty_weight * sig_wall # shape (n,)
    
    # Gradient w.r.t r
    grad[2 * n:3 * n] += factor
    
    # Gradient w.r.t x, y
    # d(wall_min)/dx_i = -w_left + w_right
    # d(wall_min)/dy_i = -w_down + w_up
    dx_wall = -weights[:, 0] + weights[:, 1]
    dy_wall = -weights[:, 2] + weights[:, 3]
    
    grad[0:n] -= factor * dx_wall
    grad[n:2 * n] -= factor * dy_wall

    # Pairwise terms
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists_sq = dx * dx + dy * dy
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero, masked later
    
    r_sum = r[:, None] + r[None, :]
    viol = r_sum - dists
    
    s = 1.0 / (1.0 + np.exp(-beta * viol))
    np.fill_diagonal(s, 0.0)
    
    # Gradient w.r.t positions
    # d(viol)/dx_i = 1 - (x_i - x_j)/dist_ij
    # We need to sum over j != i
    # Let's compute the force vector for each i
    
    # Force from j on i: factor * (dx/dx_i, dy/dy_i)
    # dx/dx_i = 1 (for viol = r_i + r_j - dist, d(dist)/dx_i = -(x_i-x_j)/dist)
    # Wait, viol = r_i + r_j - dist.
    # d(viol)/dx_i = - d(dist)/dx_i = - (-(x_i - x_j)/dist) = (x_i - x_j)/dist = dx/dists
    # So gradient of penalty w.r.t x_i is sum_j factor_ij * (x_i - x_j)/dist_ij
    
    # We have s (sigmoid of beta*viol) which is the derivative of the smooth penalty w.r.t viol.
    # d(pen)/dx_i = sum_j penalty_weight * s_ij * (x_i - x_j)/dist_ij
    
    # Create matrix of factors
    # f_ij = penalty_weight * s_ij
    f = penalty_weight * s
    
    # Gradient x_i = sum_j f_ij * (x_i - x_j)/dist_ij
    # Note that (x_i - x_j)/dist_ij is the unit vector from j to i.
    # Let u_x = dx / dists
    u_x = dx / dists
    u_y = dy / dists
    
    # Sum over j (axis=1)
    grad_x = np.sum(f * u_x, axis=1)
    grad_y = np.sum(f * u_y, axis=1)
    
    grad[0:n] += grad_x
    grad[n:2 * n] += grad_y
    
    # Gradient w.r.t r
    # d(viol)/dr_i = 1
    # grad_r_i = sum_j f_ij * 1
    grad_r = np.sum(f, axis=1)
    grad[2 * n:3 * n] += grad_r

    # Objective: -sum(r)
    grad[2 * n:3 * n] -= 1.0

    return grad

def _gradient_polish(pos, deadline, beta=40.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
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
            _soft_objective, v0, args=(n, beta),
            jac=_soft_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
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
    
    # Project to feasible
    pos_opt, r_exact = _project_feasible(pos_opt)
    
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-9):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))

def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])

def _hex_packing_init():
    """
    Hexagonal packing initialization.
    """
    pos = []
    # Approximate radius for 26 circles in hex packing
    # Area ~ 26 * pi * r^2 <= 1
    # r ~ 0.1
    # Hex lattice spacing s = 2r / sqrt(3) ?
    # Let's just place them in a hex-like grid
    rows = 5
    cols = 6
    dx = 1.0 / cols
    dy = 1.0 / rows
    offset = dx / 2
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5) * dx
            if i % 2 == 1:
                x += offset
            y = (i + 0.5) * dy
            if x < 1.0:
                pos.append([x, y])
                count += 1
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid(), _hex_packing_init()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=30000, seed=init_idx * 7 + 1,
            deadline=deadline - 25
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 20:
            break

        # Local search
        ls_pos, ls_obj = _local_search(
            sa_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 15
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Gradient-based polish (vectorized)
    # Run multiple times with different betas
    for beta in [20.0, 40.0, 80.0, 150.0]:
        if time.time() > deadline - 10:
            break
        polished = _gradient_polish(best_pos, deadline - 5, beta=beta)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 3: Final fine-tuning local search
    if time.time() < deadline - 5:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=3000, step_size=0.001,
            seed=12345, deadline=deadline - 2
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Grow radii / Final polish
    if time.time() < deadline - 2:
        grown_pos, grown_obj = _grow_radii(best_pos, max_iter=50)
        if grown_obj > best_obj:
            best_pos = grown_pos
            best_obj = grown_obj

    # Final projection to ensure validity
    final_pos, final_r = _project_feasible(best_pos)
    final_obj = float(np.sum(final_r))
    
    if not _is_valid(final_pos, eps=1e-9):
        # Fallback to a known valid configuration if projection failed
        # (Should not happen, but safe)
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    result = np.column_stack([final_pos, final_r])
    return result

# EVOLVE-BLOCK-END
