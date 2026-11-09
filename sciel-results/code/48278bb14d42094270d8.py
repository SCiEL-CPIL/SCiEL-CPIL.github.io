"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(positions):
    """
    Compute the valid packing radius for a given set of positions.
    Returns 0 if any constraint is violated (overlap or out of bounds).
    """
    if positions.shape != (N, 2) or not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check boundary constraint
    dists_origin = np.linalg.norm(positions, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
    
    # Compute pairwise distances
    # For N=50, this is 50x50, which is small enough for broadcasting
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    if min_dist <= 0:
        return 0.0
    
    r_pair = min_dist / 2.0
    r_bound = np.min(1.0 - dists_origin)
    
    return min(r_pair, r_bound)

def _penalty_score(positions):
    """
    Compute a smooth penalty score for PSO.
    We want to maximize the radius.
    Penalty for overlap: sum of (d_min - dist)^2 for overlapping pairs?
    Actually, let's use a barrier function approach.
    
    Objective: Maximize r.
    Constraints:
    1. |c_i| + r <= 1  =>  r <= 1 - |c_i|
    2. |c_i - c_j| >= 2r => r <= |c_i - c_j| / 2
    
    We can define the score as:
    S = min( min_i(1 - |c_i|), min_{i<j} |c_i - c_j|/2 )
    
    But to make it differentiable/smooth for PSO, we can use a soft penalty.
    Let R = min( min_i(1 - |c_i|), min_{i<j} |c_i - c_j|/2 )
    
    However, PSO doesn't need differentiability, it just needs a fitness landscape.
    The issue with the raw min is that it's non-smooth.
    A common trick is to use a large penalty for violations.
    
    Let's define:
    violation_boundary = max(0, max_i(|c_i| + r_est - 1)) -- but r_est is unknown.
    
    Better approach:
    We evaluate the *actual* radius r for the configuration.
    If the configuration is invalid (overlap or out of bounds), r is effectively 0 or negative?
    The task says "Score: ... computed in double precision".
    The evaluator computes the raw value.
    
    For PSO, we want to guide particles towards valid, high-radius configurations.
    If we just use `_compute_radius`, the fitness is 0 for any invalid config.
    This creates a huge plateau of 0s, making it hard for PSO to find the valid region.
    
    Instead, let's use a penalty function:
    Fitness = r_candidate - Penalty
    where r_candidate is an estimate of the radius we *could* have,
    or simply use the min-distance metric but penalize overlaps heavily.
    
    Let's try:
    1. Compute min_pair_dist and min_boundary_dist.
    2. If min_pair_dist < 2 * min_boundary_dist, the limiting factor is pair distance.
       If min_pair_dist is very small, it's an overlap.
    
    Let's use:
    score = min(min_pair_dist / 2.0, min_boundary_dist)
    But if min_pair_dist < 0 (impossible) or if points overlap such that we can't even have r=0?
    Actually, if points overlap, the "radius" is negative in a sense, but geometrically it's invalid.
    
    Let's use a penalty:
    overlap_penalty = sum( max(0, 2*0.1 - dist_ij)^2 ) ? No, we don't know the target radius.
    
    Let's stick to the raw score but add a small epsilon penalty for being close to invalid.
    Or better:
    Fitness = min(min_pair_dist / 2.0, min_boundary_dist)
    This is the same as `_compute_radius` but `_compute_radius` returns 0 if invalid.
    Wait, `_compute_radius` returns 0 if `dists_origin >= 1.0` or `min_dist <= 0`.
    If `min_dist` is small but positive, it returns a small positive number.
    So the landscape is not a flat 0. It goes to 0 as points collide or hit the wall.
    This is actually a good landscape for PSO, provided the particles don't get stuck in the "0" region.
    
    The problem with the previous PSO was likely:
    1. Too few particles or iterations.
    2. Projection step was crude.
    3. No local refinement.
    
    Let's improve the PSO:
    - Increase particles to 100.
    - Use adaptive inertia weight.
    - Add local search (Nelder-Mead) on the top 5 particles every 50 iterations.
    """
    pass

def _construct_pso():
    """Improved PSO with local refinement."""
    np.random.seed(42)
    
    n_particles = 100
    max_iter = 300
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # Initialization: Mix of random and structured
    for i in range(n_particles):
        if i % 5 == 0:
            # Structured: Hexagonal grid perturbation
            # Create a hex grid
            spacing = 0.2
            pts = []
            for row in range(-10, 11):
                for col in range(-10, 11):
                    x = (col + 0.5 * (row % 2)) * spacing
                    y = row * spacing * np.sqrt(3) / 2
                    if x**2 + y**2 < 1.0:
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= N:
                # Randomly select N points? No, this is biased.
                # Just use random for most, maybe 1 structured seed.
                pass
        
        # Random initialization
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        particles[i, :, 0] = radii * np.cos(angles)
        particles[i, :, 1] = radii * np.sin(angles)
        
        # Ensure strictly inside
        dists = np.linalg.norm(particles[i], axis=1)
        mask = dists > 0.99
        if np.any(mask):
            particles[i][mask] *= 0.99 / dists[mask][:, np.newaxis]

    # Evaluate initial fitness
    fitness = np.array([_compute_radius(particles[i]) for i in range(n_particles)])
    personal_best = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmax(fitness)
    global_best = particles[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]
    
    # Local refinement function
    def refine(positions):
        def neg_score(x_flat):
            pos = x_flat.reshape(N, 2)
            # Use a smooth version for optimization?
            # _compute_radius is non-smooth at the min.
            # Let's use a penalty based objective for local opt.
            # Maximize min(d_ij/2, 1-d_i)
            # Equivalent to: Minimize -min(...)
            # To make it smooth, we can use:
            # f(x) = - ( min_pair_dist/2 + min_boundary_dist - |min_pair_dist/2 - min_boundary_dist| ) / 2
            # This is min(a, b).
            # It's non-differentiable. L-BFGS-B might struggle.
            # Nelder-Mead is derivative-free and handles this better.
            return -_compute_radius(pos)
        
        try:
            result = minimize(neg_score, positions.flatten(), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.success or _compute_radius(result.x.reshape(N, 2)) > _compute_radius(positions):
                return result.x.reshape(N, 2)
        except Exception:
            pass
        return positions

    for iteration in range(max_iter):
        # Update inertia weight
        w = w_start - (w_start - w_end) * (iteration / max_iter)
        
        for i in range(n_particles):
            r1 = np.random.uniform(0, 1, (N, 2))
            r2 = np.random.uniform(0, 1, (N, 2))
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best[i] - particles[i]) +
                           c2 * r2 * (global_best - particles[i]))
            
            # Limit velocity to prevent overshooting
            v_max = 0.1
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] = particles[i] + velocities[i]
            
            # Soft projection: If outside, pull back towards center
            dists = np.linalg.norm(particles[i], axis=1)
            mask = dists > 0.99
            if np.any(mask):
                # Scale down to 0.99
                scale = 0.99 / dists[mask]
                particles[i][mask] *= scale[:, np.newaxis]
            
            # Check for overlaps and push apart slightly?
            # PSO doesn't do this natively. The penalty handles it.
            
            new_fitness = _compute_radius(particles[i])
            
            if new_fitness > personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = new_fitness
                
                if new_fitness > global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = new_fitness
        
        # Local refinement every 50 iterations on the best 5 particles
        if (iteration + 1) % 50 == 0:
            top_indices = np.argsort(personal_best_fitness)[::-1][:5]
            for idx in top_indices:
                refined = refine(personal_best[idx])
                ref_fitness = _compute_radius(refined)
                
                if ref_fitness > personal_best_fitness[idx]:
                    personal_best[idx] = refined
                    personal_best_fitness[idx] = ref_fitness
                    particles[idx] = refined.copy()
                    velocities[idx] = 0.0
                    
                    if ref_fitness > global_best_fitness:
                        global_best = refined.copy()
                        global_best_fitness = ref_fitness
            
            # Reseed stagnant particles
            stagnant_mask = personal_best_fitness < global_best_fitness * 0.9
            if np.any(stagnant_mask):
                for i in np.where(stagnant_mask)[0]:
                    angles = np.random.uniform(0, 2*np.pi, N)
                    radii = np.sqrt(np.random.uniform(0, 1, N))
                    particles[i, :, 0] = radii * np.cos(angles)
                    particles[i, :, 1] = radii * np.sin(angles)
                    velocities[i] = 0.0
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = _compute_radius(particles[i])
        
        # Early stopping
        if global_best_fitness > 0.1255:
            break
    
    return global_best

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    positions = _construct_pso()
    
    # Final refinement on the global best
    def neg_score(x_flat):
        pos = x_flat.reshape(N, 2)
        return -_compute_radius(pos)
    
    try:
        result = minimize(neg_score, positions.flatten(), method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if _compute_radius(result.x.reshape(N, 2)) > _compute_radius(positions):
            positions = result.x.reshape(N, 2)
    except Exception:
        pass
    
    # Final safety check and projection
    dist = np.linalg.norm(positions, axis=1)
    mask = dist > 0.9999
    if np.any(mask):
        positions[mask] *= 0.9999 / dist[mask][:, np.newaxis]
        
    return positions
# EVOLVE-BLOCK-END
