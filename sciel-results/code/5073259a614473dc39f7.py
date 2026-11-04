"""Combined program: Particle Swarm Optimization (PSO) with gradient polishing."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return 0.5 - np.max(np.abs(C))
    
    # Boundary constraint
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    # Pairwise constraint
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    # Corner points
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    # Body center points
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _energy_objective_and_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Compute violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    wall_viol = np.maximum(0.0, r_target - d_walls)
    
    energy = 0.5 * np.sum(pair_viol**2) + 0.5 * np.sum(wall_viol**2)
    
    # Gradient - vectorized
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient: for each violating pair, push apart
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol_vals = pair_viol[iu]
    
    mask = pair_viol_vals > 0
    if np.any(mask):
        i_idx = iu[0][mask]
        j_idx = iu[1][mask]
        d_vals = pair_dists[mask]
        v_vals = pair_viol_vals[mask]
        
        # Avoid division by zero
        valid = d_vals > 1e-14
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            d_v = d_vals[valid]
            v_v = v_vals[valid]
            
            vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
            coeffs = v_v / d_v  # shape (n_valid,)
            
            # Accumulate gradients
            np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
            np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    wall_viol_mask = wall_viol > 0
    if np.any(wall_viol_mask):
        i_idx = np.where(wall_viol_mask)[0]
        k_idx = np.where(wall_viol_mask)[1]
        v_vals = wall_viol[wall_viol_mask]
        
        signs = np.sign(pos[i_idx, k_idx])
        signs[signs == 0] = 1.0
        
        grad_pos[i_idx, k_idx] += v_vals * signs
    
    return energy, -grad_pos.flatten()

def _energy_minimize(pos, r_target, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for a fixed target radius."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_objective_and_grad(x, r_target)[0],
            pos.flatten(),
            jac=lambda x: _energy_objective_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 1e-8:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _soft_min_objective_and_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances (upper triangle only to save memory)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # All effective distances (pair distances divided by 2, wall distances as-is)
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -logsumexp(-beta * x) / beta
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. positions
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient
    # Vectorized pair gradient computation
    # weights shape: (n_pairs,)
    # iu[0], iu[1] shape: (n_pairs,)
    # pos[iu[0]] shape: (n_pairs, 3)
    # pos[iu[1]] shape: (n_pairs, 3)
    
    vecs = pos[iu[0]] - pos[iu[1]] # (n_pairs, 3)
    d_vals = pair_dists # (n_pairs,)
    
    # coeff = w / (2 * d)
    valid = d_vals > 1e-14
    coeffs = np.zeros_like(d_vals)
    coeffs[valid] = weights[valid] / (2.0 * d_vals[valid])
    
    # grad_pos[i] += coeff * vec
    # grad_pos[j] -= coeff * vec
    np.add.at(grad_pos, iu[0], coeffs[:, np.newaxis] * vecs)
    np.add.at(grad_pos, iu[1], -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    # wall_dists shape: (N*3,)
    # weights[len(pair_dists):] shape: (N*3,)
    w_walls = weights[len(pair_dists):]
    
    # signs
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    
    # grad_pos[i, k] -= w * sign
    grad_pos -= (w_walls.reshape(N, 3) * signs)
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    return -soft_min_val, -grad_pos.flatten()

def _soft_min_optimize(pos, beta=100.0, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Optimize using soft-min objective."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    global START_TIME
    START_TIME = time.time() # Reset timer
    
    # 1. Generate base lattice points for initialization
    V = _generate_bcc_points(4)
    
    # Scale and translate to fit in cube
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    V_extent = V_max - V_min
    max_extent = np.max(V_extent)
    
    s_init = 1.0 / (0.866 + max_extent)
    t_init = np.array([0.0, 0.0, 0.0])
    
    C_init = s_init * V + t_init
    
    # Initial polish to get a decent starting point
    C_init = _soft_min_optimize(C_init, beta=50.0, max_iter=100)
    best_r = _compute_radius(C_init)
    best_C = C_init.copy()
    
    # 2. Particle Swarm Optimization (PSO)
    # We use PSO to explore the landscape, with local polishing (soft-min) on the best particles.
    n_particles = 10
    max_iters = 50
    
    # Initialize swarm
    positions = np.zeros((n_particles, N, 3))
    velocities = np.zeros((n_particles, N, 3))
    pbest_positions = np.zeros((n_particles, N, 3))
    pbest_scores = np.zeros(n_particles)
    gbest_position = best_C.copy()
    gbest_score = best_r
    
    rng = np.random.default_rng(42)
    
    # Initialize particles around the initial solution with some noise
    for i in range(n_particles):
        noise = rng.normal(0, 0.05, size=(N, 3))
        pos_i = np.clip(best_C + noise, -HALF + 0.01, HALF - 0.01)
        positions[i] = pos_i
        # Small random initial velocities
        velocities[i] = rng.normal(0, 0.01, size=(N, 3))
        
        # Evaluate
        r_i = _compute_radius(pos_i)
        pbest_scores[i] = r_i
        pbest_positions[i] = pos_i
        
        if r_i > gbest_score:
            gbest_score = r_i
            gbest_position = pos_i.copy()
    
    # PSO Parameters
    w = 0.9  # Inertia weight
    c1 = 1.5 # Cognitive coefficient
    c2 = 1.5 # Social coefficient
    
    for it in range(max_iters):
        if _check_time():
            break
        
        for i in range(n_particles):
            if _check_time():
                break
            
            # Update velocity
            r1 = rng.random((N, 3))
            r2 = rng.random((N, 3))
            
            vel = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (gbest_position - positions[i])
            
            # Clamp velocity to prevent divergence
            max_vel = 0.1
            vel = np.clip(vel, -max_vel, max_vel)
            velocities[i] = vel
            
            # Update position
            new_pos = positions[i] + vel
            new_pos = np.clip(new_pos, -HALF + 1e-9, HALF - 1e-9)
            
            # Evaluate new position
            # To save time, we might not do a full polish every time, but the raw radius is a good proxy.
            # However, raw radius is local. Let's do a quick local polish only if it looks promising?
            # Or just evaluate raw radius for speed, and polish the gbest periodically.
            
            r_new = _compute_radius(new_pos)
            
            if r_new > pbest_scores[i]:
                pbest_scores[i] = r_new
                pbest_positions[i] = new_pos.copy()
                
                if r_new > gbest_score:
                    gbest_score = r_new
                    gbest_position = new_pos.copy()
            
            positions[i] = new_pos
        
        # Periodically polish the global best to refine it
        if it % 5 == 0:
             if not _check_time():
                 polished = _soft_min_optimize(gbest_position, beta=100.0, max_iter=100)
                 r_polished = _compute_radius(polished)
                 if r_polished > gbest_score:
                     gbest_score = r_polished
                     gbest_position = polished.copy()
                     # Update pbest for particle 0? Or just keep gbest.
                     # Let's update pbest of a random particle to the gbest to help swarm.
                     idx = rng.integers(0, n_particles)
                     pbest_positions[idx] = gbest_position
                     pbest_scores[idx] = gbest_score

    # 3. Final Polish
    # Use the best found by PSO
    final_C = gbest_position
    final_r = gbest_score
    
    # Try to push it further with a high-precision soft-min optimization
    # Start from the PSO best
    polished_C = _soft_min_optimize(final_C, beta=200.0, max_iter=500)
    polished_r = _compute_radius(polished_C)
    
    if polished_r > final_r:
        final_r = polished_r
        final_C = polished_C.copy()
    
    # One more energy minimize at the final radius to ensure no violations
    # Target slightly less than the found radius to ensure strict non-overlap
    # Actually, if we just maximize radius, we want to be as large as possible.
    # The energy minimizer minimizes violations. If violations are 0, radius is valid.
    # We can try to increase radius slightly and minimize energy.
    
    # Try a few small increments
    for _ in range(5):
        if _check_time():
            break
        target_r = final_r * 1.001
        # Add tiny noise
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(final_C + noise, -HALF + 1e-9, HALF - 1e-9)
        
        opt_pts = _energy_minimize(init_pts, target_r, max_iter=100)
        opt_r = _compute_radius(opt_pts)
        
        if opt_r > final_r:
            final_r = opt_r
            final_C = opt_pts.copy()
        else:
            break

    # Safety check: ensure validity
    if not np.all(np.isfinite(final_C)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    # Ensure inside cube (clip just in case)
    final_C = np.clip(final_C, -HALF + 1e-9, HALF - 1e-9)
    
    return final_C
