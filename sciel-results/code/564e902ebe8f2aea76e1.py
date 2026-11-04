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

def _project_feasible(pos, max_iter=200):
    """
    Iteratively adjust positions and radii to ensure strict feasibility.
    If circles overlap, push positions apart and shrink radii.
    If circles hit boundary, push inwards and shrink radii.
    """
    pos = pos.copy()
    for _ in range(max_iter):
        radii = _max_radius_for_positions(pos)
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        
        valid = True
        for i in range(N):
            # Boundary check
            min_boundary = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if radii[i] > min_boundary + 1e-12:
                # Push inwards
                if pos[i,0] < radii[i]:
                    pos[i,0] = radii[i] * 0.999 + 1e-6
                if pos[i,0] > 1.0 - radii[i]:
                    pos[i,0] = 1.0 - radii[i] * 0.999 - 1e-6
                if pos[i,1] < radii[i]:
                    pos[i,1] = radii[i] * 0.999 + 1e-6
                if pos[i,1] > 1.0 - radii[i]:
                    pos[i,1] = 1.0 - radii[i] * 0.999 - 1e-6
                valid = False
            
            # Pairwise check
            for j in range(i+1, N):
                dist = dists[i, j]
                if dist < 1e-12:
                    # Circles at same position, push apart randomly
                    pos[i,0] += 0.01
                    pos[j,0] -= 0.01
                    valid = False
                    continue
                if radii[i] + radii[j] > dist + 1e-12:
                    overlap = radii[i] + radii[j] - dist
                    dx = (pos[i,0] - pos[j,0]) / dist
                    dy = (pos[i,1] - pos[j,1]) / dist
                    push = overlap / 2.0 + 1e-6
                    pos[i,0] += dx * push
                    pos[i,1] += dy * push
                    pos[j,0] -= dx * push
                    pos[j,1] -= dy * push
                    valid = False
        
        pos = np.clip(pos, 0.0, 1.0)
        if valid:
            break
            
    final_radii = _max_radius_for_positions(pos)
    return pos, final_radii

def _soft_objective(v, n, beta=50.0):
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
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations: r_i + r_j - dist_ij
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    viol = r_sum - dists
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    viol_masked = viol[mask]
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol_masked)) / beta

    return -np.sum(r) + 60.0 * (wall_pen + pair_pen)

def _soft_gradient(v, n, beta=50.0):
    """
    Vectorized analytic gradient of _soft_objective.
    """
    x = v[0:n].copy()
    y = v[n:2 * n].copy()
    r = v[2 * n:3 * n].copy()

    grad = np.zeros(3 * n)
    penalty_weight = 60.0

    # Wall terms
    d_left = x
    d_right = 1.0 - x
    d_down = y
    d_up = 1.0 - y
    
    dists = np.stack([d_left, d_right, d_down, d_up], axis=1)
    temp = 10.0 
    exp_dists = np.exp(-dists * temp)
    sum_exp = np.sum(exp_dists, axis=1, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-12)
    weights = exp_dists / sum_exp
    
    wall_min = np.sum(weights * dists, axis=1)
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    
    factor = penalty_weight * sig_wall
    
    grad[2 * n:3 * n] += factor
    
    dx_wall = -weights[:, 0] + weights[:, 1]
    dy_wall = -weights[:, 2] + weights[:, 3]
    
    grad[0:n] -= factor * dx_wall
    grad[n:2 * n] -= factor * dy_wall

    # Pairwise terms
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists_sq = dx * dx + dy * dy
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    
    r_sum = r[:, None] + r[None, :]
    viol = r_sum - dists
    
    s = 1.0 / (1.0 + np.exp(-beta * viol))
    np.fill_diagonal(s, 0.0)
    
    f = penalty_weight * s
    
    u_x = dx / dists
    u_y = dy / dists
    
    grad_x = np.sum(f * u_x, axis=1)
    grad_y = np.sum(f * u_y, axis=1)
    
    grad[0:n] += grad_x
    grad[n:2 * n] += grad_y
    
    grad_r = np.sum(f, axis=1)
    grad[2 * n:3 * n] += grad_r

    grad[2 * n:3 * n] -= 1.0

    return grad

def _gradient_polish(pos, deadline, beta=50.0, maxiter=2000):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
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
            _soft_objective, v0, args=(n, beta),
            jac=_soft_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
        )
    except Exception:
        return pos, r_init, float(np.sum(r_init))

    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]
    r_opt = v_opt[2 * n:3 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    
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

def _initial_random(seed=0, min_dist=0.05):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 10000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > min_dist for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)

def _repulsion_relax(pos, n_iter=5000, seed=42, deadline=None):
    """
    Relax positions using repulsion forces to spread circles evenly.
    This helps avoid local minima where circles are clustered.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    n = len(current_pos)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute repulsion forces
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i+1, n):
                dx = current_pos[i,0] - current_pos[j,0]
                dy = current_pos[i,1] - current_pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dx = rng.uniform(-0.01, 0.01)
                    dy = rng.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                # Repulsion strength decreases with distance
                rep = 0.001 / (dist * dist + 0.001)
                forces[i,0] += rep * dx / dist
                forces[i,1] += rep * dy / dist
                forces[j,0] -= rep * dx / dist
                forces[j,1] -= rep * dy / dist
        
        # Add attraction to center to keep circles in square
        center = np.array([0.5, 0.5])
        for i in range(n):
            to_center = center - current_pos[i]
            forces[i] += 0.0001 * to_center
        
        # Update positions
        learning_rate = 0.1 * (1.0 - it / n_iter) + 0.01
        current_pos += learning_rate * forces
        current_pos = np.clip(current_pos, 0.01, 0.99)
        
    return current_pos

def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=50000, seed=42, deadline=None):
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

def _local_search(pos, n_iter=5000, step_size=0.008, seed=42, deadline=None):
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
            for _ in range(20):
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

def _grow_radii(pos, max_iter=100):
    """
    Try to increase radii by solving for optimal radii given fixed positions.
    This is a placeholder; in practice, radii are determined by positions.
    """
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    rng = np.random.RandomState(42)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for _ in range(15):
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid(), _hex_packing_init()]
    for s in range(8):
        initializations.append(_initial_random(seed=s * 997 + 13, min_dist=0.06))

    best_pos = None
    best_obj = -np.inf

    # Phase 0: Repulsion relaxation for better initial structures
    relaxed_inits = []
    for init_pos in initializations:
        if time.time() > deadline - 60:
            break
        relaxed = _repulsion_relax(init_pos, n_iter=2000, seed=hash(init_pos) % 1000, deadline=deadline - 55)
        relaxed_inits.append(relaxed)

    # Phase 1: Simulated annealing + local search from each init
    for init_idx, init_pos in enumerate(relaxed_inits):
        if time.time() > deadline - 40:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=40000, seed=init_idx * 7 + 1,
            deadline=deadline - 35
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 25:
            break

        # Local search
        ls_pos, ls_obj = _local_search(
            sa_pos, n_iter=3000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 20
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
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 15:
            break
        polished = _gradient_polish(best_pos, deadline - 10, beta=beta, maxiter=3000)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 3: Final fine-tuning local search
    if time.time() < deadline - 8:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=5000, step_size=0.001,
            seed=12345, deadline=deadline - 5
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Grow radii / Final polish
    if time.time() < deadline - 3:
        grown_pos, grown_obj = _grow_radii(best_pos, max_iter=100)
        if grown_obj > best_obj:
            best_pos = grown_pos
            best_obj = grown_obj

    # Final projection to ensure validity
    final_pos, final_r = _project_feasible(best_pos)
    final_obj = float(np.sum(final_r))
    
    if not _is_valid(final_pos, eps=1e-9):
        # Fallback to a known valid configuration if projection failed
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    result = np.column_stack([final_pos, final_r])
    return result

# EVOLVE-BLOCK-END
