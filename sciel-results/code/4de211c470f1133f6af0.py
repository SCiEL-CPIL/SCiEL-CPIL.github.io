import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def _soft_repulsion_force(points, r_target, k_rep=50.0, k_bound=100.0):
    """
    Compute smooth repulsive forces for particles closer than r_target,
    plus boundary repulsion for particles too close to the container wall.
    Uses a soft potential that vanishes at contact distance.
    """
    forces = np.zeros_like(points)
    
    # Pairwise repulsion: force = k_rep * (r_target - dist) * direction if dist < r_target
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Only compute forces for pairs closer than r_target
    mask = dists < r_target
    # For each pair, compute the force contribution
    # F_ij = k_rep * (r_target - d_ij) * (r_i - r_j) / d_ij
    for i in range(N):
        for j in range(i+1, N):
            if mask[i, j]:
                d = dists[i, j]
                if d > 1e-10:
                    force_mag = k_rep * (r_target - d)
                    force_vec = force_mag * diff[i, j] / d
                    forces[i] += force_vec
                    forces[j] -= force_vec
    
    # Boundary repulsion: if |r_i| > 1 - r_target, push inward
    rad = np.linalg.norm(points, axis=1)
    boundary_mask = rad > (1.0 - r_target)
    for i in range(N):
        if boundary_mask[i]:
            r = rad[i]
            if r > 1e-10:
                force_mag = k_bound * (r - (1.0 - r_target))
                force_vec = -force_mag * points[i] / r
                forces[i] += force_vec
    
    return forces

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def _dynamics_evolve(points, time_budget, dt=0.001, damping=0.98, 
                     r_target_init=0.053, r_target_max=0.065,
                     k_rep=50.0, k_bound=100.0):
    """
    Evolve the packing using symplectic (leapfrog) integration with
    smooth repulsive potentials. Particles repel each other when closer
    than r_target, and are repelled from the boundary.
    
    This is a structural change: instead of gradient-based optimization,
    we use physical dynamics to naturally resolve overlaps and boundary violations.
    """
    start_time = time.time()
    points = points.copy()
    velocities = np.zeros_like(points)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    r_target = r_target_init
    step_count = 0
    
    while time.time() - start_time < time_budget:
        # Compute forces
        forces = _soft_repulsion_force(points, r_target, k_rep, k_bound)
        
        # Leapfrog integration (symplectic)
        # v_half = v + 0.5 * dt * F
        velocities_half = velocities + 0.5 * dt * forces
        # x_new = x + dt * v_half
        points_new = points + dt * velocities_half
        # v_new = v_half + 0.5 * dt * F_new (approximate using same forces for stability)
        velocities_new = velocities_half + 0.5 * dt * forces
        
        # Apply damping
        velocities_new *= damping
        
        # Project to unit circle
        rad = np.linalg.norm(points_new, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            points_new[mask] /= rad[mask, np.newaxis]
        
        points = points_new
        velocities = velocities_new
        step_count += 1
        
        # Periodically evaluate and update best
        if step_count % 100 == 0:
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
                # Gradually increase r_target to encourage larger packing
                r_target = min(r_target + 0.0005, r_target_max)
        
        # Check for convergence (low kinetic energy)
        if step_count % 500 == 0:
            ke = 0.5 * np.sum(velocities**2)
            if ke < 1e-10:
                # System has settled; try a small perturbation to escape local minimum
                noise = np.random.normal(0, 0.005, size=points.shape)
                points += noise
                velocities = np.zeros_like(points)
                rad = np.linalg.norm(points, axis=1)
                mask = rad > 1.0
                if np.any(mask):
                    points[mask] /= rad[mask, np.newaxis]
    
    return best_points, best_score

def simulated_annealing(points, time_budget, initial_temp=0.02, cooling_rate=0.98, step_size=0.01):
    """
    Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = initial_temp
    
    while time.time() - start_time < time_budget:
        # Generate perturbation
        num_moves = np.random.randint(5, 20)
        indices = np.random.choice(N, num_moves, replace=False)
        
        proposed_points = current_points.copy()
        noise = np.random.normal(0, step_size, size=(num_moves, 2))
        proposed_points[indices] += noise
        
        # Project to unit circle
        rad = np.linalg.norm(proposed_points, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            proposed_points[mask] /= rad[mask, np.newaxis]
            
        proposed_score = compute_score(proposed_points)
        
        # Acceptance criterion
        delta = proposed_score - current_score
        if delta > 0:
            current_points = proposed_points
            current_score = proposed_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_points = proposed_points
                current_score = proposed_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        # Cool down
        T *= cooling_rate
        if T < 1e-5:
            T = 1e-5
            
    return best_points, best_score

def polish_least_squares(points, time_budget, k=100.0, R_target=0.06):
    """
    Polish using scipy.optimize.least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container to ensure validity during optimization
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        # Use trf (trust-region reflective)
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=20000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        return points, compute_score(points)

def polish_lbfgsb(points, time_budget, k=100.0):
    """
    Polish using scipy.optimize.minimize with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        # We want to maximize s_val, so minimize -s_val
        return -s_val
    
    x0 = points.flatten()
    
    try:
        # Use L-BFGS-B with bounds to keep points roughly inside
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12}
        )
        
        x_opt = result.x.reshape(N, 2)
        
        # Project inside
        rad = np.linalg.norm(x_opt, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_opt[mask] /= rad[mask, np.newaxis]
        
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        return points, compute_score(points)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Optimization Phase
    # Phase 1: Dynamics-based evolution (structural change: rigid-body dynamics with soft potentials)
    dyn_budget = 40.0
    dyn_elapsed = time.time() - start_time
    if time_limit - dyn_elapsed > dyn_budget:
        dyn_points, dyn_score = _dynamics_evolve(best_points, dyn_budget)
        if dyn_score > best_score:
            best_score = dyn_score
            best_points = dyn_points
    
    # Phase 2: Simulated Annealing to escape local minima found by dynamics
    sa_budget = min(25.0, time_limit - (time.time() - start_time) - 10.0)
    if sa_budget > 5.0:
        sa_points, sa_score = simulated_annealing(best_points, sa_budget)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Phase 3: L-BFGS-B polish
    lbfgsb_budget = min(15.0, time_limit - (time.time() - start_time) - 5.0)
    if lbfgsb_budget > 5.0:
        polished_points, polished_score = polish_lbfgsb(best_points, lbfgsb_budget)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
    
    # Phase 4: Final least_squares polish
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 5.0:
        polished_points, polished_score = polish_least_squares(best_points, remaining - 3.0)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
    
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
