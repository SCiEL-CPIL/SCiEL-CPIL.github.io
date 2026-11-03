"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(500):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-10
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _interior_point_polish(xyz, time_budget=8.0):
    """
    Log-barrier interior-point style polish.
    Minimizes -sum(r) + mu * sum(-log(slack_k)) where slack_k >= 0 for each constraint.
    This keeps the solution strictly feasible and avoids the penalty method's tendency
    to violate constraints slightly.
    """
    start_time = time.time()
    
    x = xyz[:, 0].copy().astype(float)
    y = xyz[:, 1].copy().astype(float)
    r = xyz[:, 2].copy().astype(float)
    
    # Ensure initial feasibility with margin
    for i in range(N):
        r[i] = min(r[i], x[i] - 1e-4, 1 - x[i] - 1e-4, y[i] - 1e-4, 1 - y[i] - 1e-4)
        r[i] = max(r[i], 1e-6)
        x[i] = np.clip(x[i], r[i] + 1e-4, 1 - r[i] - 1e-4)
        y[i] = np.clip(y[i], r[i] + 1e-4, 1 - r[i] - 1e-4)
    
    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    def _log_barrier_obj(params, mu):
        """
        Log-barrier objective: -sum(r) - mu * sum(log(slack_k))
        Slack variables:
          - x_i - r_i (left x wall)
          - 1 - x_i - r_i (right x wall)
          - y_i - r_i (left y wall)
          - 1 - y_i - r_i (right y wall)
          - dist_ij - r_i - r_j (overlap)
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Clamp to avoid log of non-positive
        margin = 1e-12
        
        # Boundary slacks
        s_lx = x - r
        s_ux = 1 - x - r
        s_ly = y - r
        s_uy = 1 - y - r
        
        # Overlap slacks
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        s_ov = dist - r[idx_i] - r[idx_j]
        
        # Sum of log of positive parts
        # Use softplus-like: log(max(s, margin))
        log_s_lx = np.log(np.maximum(s_lx, margin))
        log_s_ux = np.log(np.maximum(s_ux, margin))
        log_s_ly = np.log(np.maximum(s_ly, margin))
        log_s_uy = np.log(np.maximum(s_uy, margin))
        log_s_ov = np.log(np.maximum(s_ov, margin))
        
        val = -np.sum(r) - mu * (np.sum(log_s_lx) + np.sum(log_s_ux) +
                                  np.sum(log_s_ly) + np.sum(log_s_uy) +
                                  np.sum(log_s_ov))
        
        return val
    
    def _log_barrier_grad(params, mu):
        """Gradient of log-barrier objective."""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # d(-sum r)/dr
        gr -= 1.0
        
        margin = 1e-12
        
        # Boundary slacks
        s_lx = x - r
        s_ux = 1 - x - r
        s_ly = y - r
        s_uy = 1 - y - r
        
        # Gradient of -mu * log(s) = -mu / s * ds/dvar
        # s_lx = x - r: ds/dx = 1, ds/dr = -1
        safe_s_lx = np.maximum(s_lx, margin)
        gx += -mu / safe_s_lx
        gr += mu / safe_s_lx
        
        safe_s_ux = np.maximum(s_ux, margin)
        gx += mu / safe_s_ux
        gr += mu / safe_s_ux
        
        safe_s_ly = np.maximum(s_ly, margin)
        gy += -mu / safe_s_ly
        gr += mu / safe_s_ly
        
        safe_s_uy = np.maximum(s_uy, margin)
        gy += mu / safe_s_uy
        gr += mu / safe_s_uy
        
        # Overlap slacks
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        s_ov = dist - r[idx_i] - r[idx_j]
        safe_s_ov = np.maximum(s_ov, margin)
        
        inv_dist = 1.0 / dist
        factor = -mu / safe_s_ov
        
        # ds_ov/dx_i = dx/dist, ds_ov/dx_j = -dx/dist
        np.add.at(gx, idx_i, factor * dx * inv_dist)
        np.add.at(gx, idx_j, -factor * dx * inv_dist)
        np.add.at(gy, idx_i, factor * dy * inv_dist)
        np.add.at(gy, idx_j, -factor * dy * inv_dist)
        np.add.at(gr, idx_i, -factor)
        np.add.at(gr, idx_j, -factor)
        
        grad = np.empty_like(params)
        grad[0::3] = gx
        grad[1::3] = gy
        grad[2::3] = gr
        
        return grad
    
    # Multi-scale mu annealing
    best_params = np.concatenate([x, y, r]).copy()
    best_val = _log_barrier_obj(best_params, 10.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for mu in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _log_barrier_obj(p, mu)
        
        def grad(p):
            return _log_barrier_grad(p, mu)
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14,
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val = _log_barrier_obj(result.x, mu)
            if obj_val < best_val:
                best_params = result.x.copy()
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B with multi-scale penalty.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    # Multi-scale penalty annealing
    for penalty_weight in [500.0, 2000.0, 10000.0, 50000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-15, 
                                      'gtol': 1e-13, 'maxfun': 15000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x.copy()
                best_val = obj_val
        except Exception:
            pass
    
    # Final tight polish
    if time.time() - start_time < time_budget:
        penalty_weight = 100000.0
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight, smooth_eps=1e-5)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight, smooth_eps=1e-5)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 5000, 'ftol': 1e-16, 
                                      'gtol': 1e-14, 'maxfun': 20000})
            obj_val, _ = _objective_grad(result.x, penalty_weight, smooth_eps=1e-5)
            if obj_val < best_val:
                best_params = result.x.copy()
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _physics_relax(xyz, time_limit=3.0, grow_rate=0.0015):
    """
    Force-directed physics relaxation with adaptive forces and radius growth.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.004
    velocity_x = np.zeros(N)
    velocity_y = np.zeros(N)
    damping = 0.7
    
    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                
                # Repulsion if overlapping
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Strong repulsion proportional to overlap
                    force = 100.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 100.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= 100.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += 100.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= 100.0 * (y[i] - (1 - r[i]))
        
        # Update velocities with damping
        velocity_x = velocity_x * damping + dt * fx
        velocity_y = velocity_y * damping + dt * fy
        
        # Update positions
        x += velocity_x
        y += velocity_y
        
        # Clip positions
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Update xyz
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = xyz.copy()
    
    return best_xyz


def _greedy_placement():
    """
    Greedy largest-first placement: place circles one by one at the position
    that allows the largest possible radius.
    """
    xyz = np.zeros((N, 3))
    placed = np.zeros(N, dtype=bool)
    
    for idx in range(N):
        best_pos = (0.5, 0.5)
        best_r = 0.0
        
        # Try a grid of candidate positions
        for cx in np.linspace(0.02, 0.98, 20):
            for cy in np.linspace(0.02, 0.98, 20):
                # Max radius from walls
                r_wall = min(cx, 1 - cx, cy, 1 - cy)
                if r_wall <= 0.005:
                    continue
                
                # Check against placed circles
                min_dist = np.inf
                for j in range(idx):
                    if not placed[j]:
                        continue
                    dx = cx - xyz[j, 0]
                    dy = cy - xyz[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    max_r = max(0, d - xyz[j, 2])
                    min_dist = min(min_dist, max_r)
                
                r_candidate = min(r_wall, min_dist)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_pos = (cx, cy)
        
        xyz[idx, 0] = best_pos[0]
        xyz[idx, 1] = best_pos[1]
        xyz[idx, 2] = max(best_r, 0.01)
        placed[idx] = True
    
    return xyz


def _corner_anchored():
    """
    Place large circles in corners and edges, then fill interior.
    """
    xyz = np.zeros((N, 3))
    
    # Four corners
    corner_r = 0.15
    xyz[0] = [corner_r, corner_r, corner_r]
    xyz[1] = [1 - corner_r, corner_r, corner_r]
    xyz[2] = [corner_r, 1 - corner_r, corner_r]
    xyz[3] = [1 - corner_r, 1 - corner_r, corner_r]
    
    # Edge midpoints
    edge_r = 0.10
    xyz[4] = [0.5, edge_r, edge_r]
    xyz[5] = [0.5, 1 - edge_r, edge_r]
    xyz[6] = [edge_r, 0.5, edge_r]
    xyz[7] = [1 - edge_r, 0.5, edge_r]
    
    # Interior grid
    remaining = N - 8
    interior_positions = []
    # 4x4 interior grid
    for i in range(4):
        for j in range(4):
            x = 0.25 + i * 0.1667
            y = 0.25 + j * 0.1667
            interior_positions.append([x, y])
    
    interior_positions = interior_positions[:remaining]
    for i, pos in enumerate(interior_positions):
        xyz[8 + i] = [pos[0], pos[1], 0.06]
    
    return xyz


def _hex_initial():
    """Hexagonal packing initial configuration."""
    positions = []
    row_height = 0.155
    for row in range(7):
        y = 0.075 + row * row_height
        if row % 2 == 0:
            for col in range(5):
                x = 0.1 + col * 0.2
                positions.append([x, y])
        else:
            for col in range(5):
                x = 0.2 + col * 0.2
                positions.append([x, y])
        if len(positions) >= N:
            break
    positions = positions[:N]
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.075
    return np.column_stack([positions, radii])


def _grid_initial():
    """5x5 grid + 1 center initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions.append([0.5, 0.5])
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.08
    return np.column_stack([positions, radii])


def _random_initial(seed):
    """Random initial configuration."""
    np.random.seed(seed)
    positions = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.ones(N, dtype=np.float64) * 0.05
    return np.column_stack([positions, radii])


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses multi-start strategy with diverse initializations, physics relaxation,
    and multi-scale analytic polishing plus interior-point polish.
    """
    start_time = time.time()
    deadline = _TIME_LIMIT - 5.0  # safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Corner-anchored (exploits known structure)
    configs.append(_corner_anchored())
    
    # 2. Greedy largest-first
    configs.append(_greedy_placement())
    
    # 3. Hexagonal
    configs.append(_hex_initial())
    
    # 4. Grid
    configs.append(_grid_initial())
    
    # 5-8. Random with different seeds
    for seed in [42, 123, 777, 2024]:
        configs.append(_random_initial(seed))
    
    # Process each configuration with time management
    num_configs = len(configs)
    time_per_config = (deadline - _elapsed()) / num_configs
    time_per_config = max(time_per_config, 2.0)  # minimum 2s per config
    
    for config_idx, xyz in enumerate(configs):
        if _elapsed() > deadline - 15.0:  # reserve 15s for final polish
            break
        
        config_time = min(time_per_config, deadline - _elapsed() - 15.0)
        if config_time < 1.0:
            break
        
        # Split time: 30% physics, 40% penalty polish, 30% interior-point polish
        physics_time = config_time * 0.30
        penalty_time = config_time * 0.40
        ip_time = config_time * 0.30
        
        # Physics relaxation
        xyz_relaxed = _physics_relax(xyz, time_limit=physics_time)
        
        # Polish with analytic gradient (penalty method)
        xyz_polished = _polish_analytic(xyz_relaxed, time_budget=penalty_time)
        
        # Interior-point polish
        xyz_ip = _interior_point_polish(xyz_polished, time_budget=ip_time)
        
        # Enforce validity for both
        xyz_final_p = _enforce_validity(xyz_polished)
        xyz_final_ip = _enforce_validity(xyz_ip)
        
        score_p = np.sum(xyz_final_p[:, 2])
        score_ip = np.sum(xyz_final_ip[:, 2])
        
        if score_p > best_score:
            best_score = score_p
            best_result = xyz_final_p.copy()
        
        if score_ip > best_score:
            best_score = score_ip
            best_result = xyz_final_ip.copy()
    
    # Final dedicated polish for the best candidate
    if best_result is not None and _elapsed() < deadline - 5.0:
        remaining_time = deadline - _elapsed() - 5.0
        if remaining_time > 3.0:
            # Try both penalty and interior-point on the best
            half_time = remaining_time / 2.0
            
            xyz_p1 = _polish_analytic(best_result, time_budget=half_time)
            xyz_p1 = _enforce_validity(xyz_p1)
            score_p1 = np.sum(xyz_p1[:, 2])
            if score_p1 > best_score:
                best_score = score_p1
                best_result = xyz_p1
            
            xyz_p2 = _interior_point_polish(best_result, time_budget=half_time)
            xyz_p2 = _enforce_validity(xyz_p2)
            score_p2 = np.sum(xyz_p2[:, 2])
            if score_p2 > best_score:
                best_score = score_p2
                best_result = xyz_p2
    
    # If no valid result, use the first config as fallback
    if best_result is None:
        best_result = _enforce_validity(configs[0])
    
    # Final validation and cleanup
    for i in range(N):
        best_result[i, 0] = np.clip(best_result[i, 0], 0, 1)
        best_result[i, 1] = np.clip(best_result[i, 1], 0, 1)
        max_r = min(best_result[i, 0], best_result[i, 1], 
                   1 - best_result[i, 0], 1 - best_result[i, 1])
        best_result[i, 2] = min(best_result[i, 2], max_r)
        best_result[i, 2] = max(best_result[i, 2], 1e-6)
    
    return best_result
# EVOLVE-BLOCK-END
