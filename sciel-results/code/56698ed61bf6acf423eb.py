"""
50 equal circles in a unit square – combined contact-graph + soft-min optimization.

Strategy:
  1. Start from a hexagonal grid.
  2. Use a multi-stage L-BFGS soft-minimization with adaptive tau annealing.
  3. Interleave with simulated annealing that uses the actual radius as objective.
  4. Use contact-graph rigid-body solving to snap to exact contacts.
  5. Final high-precision polish.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)


def _fast_radius(centers):
    """Fast radius computation using only nearest neighbors via KDTree."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)


def _soft_min_objective(x, tau=0.005):
    """Objective: minimize -soft_min(minimum clearance)."""
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min


def _grad_soft_min(x, tau=0.005):
    """Analytic gradient of -soft_min."""
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    iu, ju = np.triu_indices(N, k=1)
    diff = centers[iu] - centers[ju]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    for i in range(N):
        w = weights[i]
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad[2*i] += w * np.sign(centers[i, 0])
        if wall_dists_y[i] <= wall_dists_x[i]:
            grad[2*i+1] += w * np.sign(centers[i, 1])
    
    # Pair gradients
    for k in range(len(iu)):
        w = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[k]
        if d > 1e-12:
            factor = 0.5 * w / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad


def _build_contact_graph(centers, r, margin=0.015):
    """Build contact graph: edges where distance ≈ 2r, walls where dist ≈ r."""
    edges = []
    walls = []
    
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2 * r + margin)
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if abs(d - 2 * r) < margin:
            edges.append((i, j))
    
    for i in range(N):
        x, y = centers[i]
        if abs(0.5 - x - r) < margin:
            walls.append((i, 0, 1))
        if abs(-0.5 + x + r) < margin:
            walls.append((i, 0, -1))
        if abs(0.5 - y - r) < margin:
            walls.append((i, 1, 1))
        if abs(-0.5 + y + r) < margin:
            walls.append((i, 1, -1))
    
    return edges, walls


def _solve_rigid_assembly(centers, r, edges, walls, max_iter=80, tol=1e-12):
    """Solve for positions satisfying contact constraints via iterative relaxation."""
    pos = centers.copy().astype(np.float64)
    
    for _ in range(max_iter):
        max_change = 0.0
        
        for (i, axis, sign) in walls:
            if axis == 0:
                target = (0.5 - r) if sign == 1 else (-0.5 + r)
                change = target - pos[i, 0]
                pos[i, 0] += change
                max_change = max(max_change, abs(change))
            else:
                target = (0.5 - r) if sign == 1 else (-0.5 + r)
                change = target - pos[i, 1]
                pos[i, 1] += change
                max_change = max(max_change, abs(change))
        
        for (i, j) in edges:
            diff = pos[i] - pos[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            error = (d - 2 * r) / d
            correction = 0.5 * error * diff
            pos[i] -= correction
            pos[j] += correction
            max_change = max(max_change, 0.5 * abs(error) * d)
        
        if max_change < tol:
            break
    
    return pos


def _contact_energy(centers, r, edges, walls):
    """Total constraint violation energy."""
    energy = 0.0
    for (i, j) in edges:
        d = np.linalg.norm(centers[i] - centers[j])
        energy += 0.5 * (d - 2 * r) ** 2
    for (i, axis, sign) in walls:
        if axis == 0:
            err = (0.5 - centers[i, 0]) - r if sign == 1 else (centers[i, 0] + 0.5) - r
        else:
            err = (0.5 - centers[i, 1]) - r if sign == 1 else (centers[i, 1] + 0.5) - r
        energy += 0.5 * err ** 2
    return energy


def _anneal_contacts(centers, r, rng, time_budget):
    """Simulated annealing on contact-graph energy with periodic rigid solving."""
    start_time = time.time()
    pos = centers.copy().astype(np.float64)
    best = pos.copy()
    best_r = _compute_radius(pos)
    
    T = 0.004
    T_min = 0.00005
    n_steps = 30000
    rebuild_interval = 150
    
    edges, walls = _build_contact_graph(pos, r, margin=0.02)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = T_min + (T - T_min) * (1 - step / n_steps)
        
        if step % rebuild_interval == 0:
            edges, walls = _build_contact_graph(pos, r, margin=0.02)
        
        E_curr = _contact_energy(pos, r, edges, walls)
        
        move_size = 0.015 * (T / T_min) ** 0.3 + 0.003
        idx = rng.integers(0, N)
        axis = rng.integers(0, 2)
        delta = rng.normal(0, move_size)
        
        pos_new = pos.copy()
        pos_new[idx, axis] += delta
        pos_new = np.clip(pos_new, -0.5, 0.5)
        
        E_new = _contact_energy(pos_new, r, edges, walls)
        dE = E_new - E_curr
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            pos = pos_new
        
        if step % rebuild_interval == 0:
            pos = _solve_rigid_assembly(pos, r, edges, walls, max_iter=40)
            pos = np.clip(pos, -0.5, 0.5)
            
            r_est = _compute_radius(pos)
            if r_est > r * 0.95:
                r = min(r_est, r * 1.01)
            
            r_curr = _compute_radius(pos)
            if r_curr > best_r:
                best_r = r_curr
                best = pos.copy()
    
    edges, walls = _build_contact_graph(pos, r, margin=0.02)
    pos = _solve_rigid_assembly(pos, r, edges, walls, max_iter=100)
    pos = np.clip(pos, -0.5, 0.5)
    r_curr = _compute_radius(pos)
    if r_curr > best_r:
        best_r = r_curr
        best = pos.copy()
    
    return best, best_r


def _anneal_radius(centers, rng, time_budget):
    """Simulated annealing using actual radius as the objective."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.015
    T_end = 1e-9
    max_steps = 400000
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        idx = rng.integers(0, N)
        step_size = 0.008 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -4 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r


def _lbfgs_polish(x0, tau, time_budget):
    """L-BFGS-B polish on soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_objective(x, tau), _grad_soft_min(x, tau)
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 1200, 'disp': False}
        )
        return np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)


def _initial_hex_grid():
    """Hexagonal grid initial placement."""
    rows = 8
    cols = 7
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for row in range(rows):
        y = -0.5 + dy * (row + 0.5)
        if y > 0.5:
            break
        offset = dx / 2 if row % 2 == 1 else 0
        for col in range(cols):
            x = -0.5 + dx * (col + 0.5) + offset
            if -0.5 <= x <= 0.5:
                points.append([x, y])
                if len(points) >= N:
                    break
        if len(points) >= N:
            break
    
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for y in ticks:
            for x in ticks:
                if len(points) >= N:
                    break
                if not any(np.allclose([x, y], p) for p in points):
                    points.append([x, y])
            if len(points) >= N:
                break
    
    grid = np.array(points[:N])
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.002, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def _initial_grid_random(seed):
    """Random initial placement with minimum separation."""
    rng = np.random.default_rng(seed)
    attempts = 0
    max_attempts = 50000
    points = []
    
    while len(points) < N and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(-0.45, 0.45)
        y = rng.uniform(-0.45, 0.45)
        
        ok = True
        for px, py in points:
            if (x - px)**2 + (y - py)**2 < (0.08)**2:
                ok = False
                break
        
        if ok:
            points.append([x, y])
    
    while len(points) < N:
        points.append([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45)])
    
    return np.array(points[:N])


def _anneal_with_kdtree(centers, rng, time_budget):
    """Faster simulated annealing using KDTree for radius computation."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.012
    T_end = 1e-10
    max_steps = 500000
    
    tree = cKDTree(current)
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        idx = rng.integers(0, N)
        step_size = 0.006 * (1.0 + 0.2 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        # Fast radius check using KDTree
        new_r = _fast_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            tree = cKDTree(current)
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -3 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
                    tree = cKDTree(current)
    
    return best, best_r


def _multi_start_optimize(centers, rng, time_budget):
    """Try multiple perturbations and keep the best."""
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    n_starts = 8
    per_budget = time_budget / n_starts
    
    for i in range(n_starts):
        if time.time() - start_time > time_budget - 2:
            break
        
        # Perturb the best solution
        perturb_size = 0.01 * (1 + i * 0.3)
        perturbed = best + rng.normal(0, perturb_size, best.shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Quick polish
        tb = min(per_budget, deadline - time.time() - 1)
        if tb > 1:
            polished = _lbfgs_polish(perturbed.flatten(), 0.005, tb)
            r = _compute_radius(polished)
            if r > best_r:
                best_r = r
                best = polished
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centers = _initial_hex_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Coarse soft-min L-BFGS
    if time.time() < deadline - 60:
        x0 = best_centers.flatten()
        for tau in [0.03, 0.015, 0.008]:
            if time.time() > deadline - 55:
                break
            tb = min(10.0, deadline - time.time() - 50)
            if tb < 3:
                break
            trial = _lbfgs_polish(x0, tau, tb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 2: Contact-graph annealing
    if time.time() < deadline - 45:
        sa_budget = min(25.0, deadline - time.time() - 40)
        if sa_budget > 5:
            best_centers, best_radius = _anneal_contacts(
                best_centers, best_radius, rng, sa_budget
            )
    
    # Phase 3: Fast KDTree-based radius annealing
    if time.time() < deadline - 30:
        sa_budget = min(20.0, deadline - time.time() - 25)
        if sa_budget > 5:
            best_centers, best_radius = _anneal_with_kdtree(
                best_centers, rng, sa_budget
            )
    
    # Phase 4: Fine soft-min L-BFGS
    if time.time() < deadline - 20:
        x0 = best_centers.flatten()
        for tau in [0.005, 0.003, 0.002, 0.001]:
            if time.time() > deadline - 15:
                break
            tb = min(8.0, deadline - time.time() - 12)
            if tb < 2:
                break
            trial = _lbfgs_polish(x0, tau, tb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 5: Multi-start perturbation
    if time.time() < deadline - 15:
        ms_budget = min(12.0, deadline - time.time() - 12)
        if ms_budget > 4:
            best_centers, best_radius = _multi_start_optimize(
                best_centers, rng, ms_budget
            )
    
    # Phase 6: Another contact-graph refinement
    if time.time() < deadline - 10:
        sa_budget = min(8.0, deadline - time.time() - 8)
        if sa_budget > 3:
            best_centers, best_radius = _anneal_contacts(
                best_centers, best_radius, rng, sa_budget
            )
    
    # Phase 7: Final high-precision polish
    if time.time() < deadline - 5:
        tb = min(8.0, deadline - time.time() - 4)
        if tb > 2:
            trial = _lbfgs_polish(best_centers.flatten(), 0.0005, tb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Phase 8: Micro-polish
    if time.time() < deadline - 2:
        tb = min(4.0, deadline - time.time() - 1)
        if tb > 1:
            try:
                result = opt.minimize(
                    lambda v: _soft_min_objective(v, tau=0.0003),
                    best_centers.flatten(), method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-18, 'gtol': 1e-16,
                             'maxfun': 400, 'disp': False}
                )
                polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished
            except Exception:
                pass
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hex_grid()
    
    return best_centers
