"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _raw_score(positions):
    """
    Compute the raw packing radius (can be negative if invalid).
    This provides a smooth-ish landscape for PSO.
    """
    if positions.shape != (N, 2) or not np.all(np.isfinite(positions)):
        return -np.inf
    
    dists_origin = np.linalg.norm(positions, axis=1)
    r_bound = np.min(1.0 - dists_origin)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r_pair = min_dist / 2.0
    
    return min(r_pair, r_bound)

def _generate_hex_grid(spacing=0.25):
    """Generate a hexagonal grid of points inside the unit circle."""
    points = []
    # Estimate range
    max_coord = int(1.0 / spacing) + 2
    for row in range(-max_coord, max_coord + 1):
        for col in range(-max_coord, max_coord + 1):
            x = (col + 0.5 * (row % 2)) * spacing
            y = row * spacing * np.sqrt(3) / 2
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
    return np.array(points)

def _init_particle():
    """Initialize a particle using a perturbed hexagonal lattice."""
    # Generate a hex grid with spacing suitable for ~50 points
    # Spacing 0.25 yields ~58 points in unit circle
    grid = _generate_hex_grid(0.25)
    
    if len(grid) < N:
        # Fallback to random if grid is too sparse (should not happen)
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    else:
        # Shuffle and take first N
        indices = np.random.choice(len(grid), N, replace=False)
        pos = grid[indices].copy()
        
        # Add small random jitter to break symmetry
        pos += np.random.uniform(-0.02, 0.02, (N, 2))
        
    # Project to unit circle
    dists = np.linalg.norm(pos, axis=1)
    mask = dists > 0.99
    if np.any(mask):
        pos[mask] *= 0.99 / dists[mask][:, np.newaxis]
        
    return pos

def _construct_pso():
    """PSO with hex initialization and local refinement."""
    np.random.seed(42)
    start_time = time.time()
    time_limit = 90.0
    
    n_particles = 60
    max_iter = 200
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    for i in range(n_particles):
        particles[i] = _init_particle()
        
    fitness = np.array([_raw_score(particles[i]) for i in range(n_particles)])
    personal_best = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmax(fitness)
    global_best = particles[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]
    
    def refine(positions):
        def neg_score(x_flat):
            pos = x_flat.reshape(N, 2)
            return -_raw_score(pos)
        
        try:
            result = minimize(neg_score, positions.flatten(), method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            new_pos = result.x.reshape(N, 2)
            if _raw_score(new_pos) > _raw_score(positions):
                return new_pos
        except Exception:
            pass
        return positions

    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        w = w_start - (w_start - w_end) * (iteration / max_iter)
        
        for i in range(n_particles):
            r1 = np.random.uniform(0, 1, (N, 2))
            r2 = np.random.uniform(0, 1, (N, 2))
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best[i] - particles[i]) +
                           c2 * r2 * (global_best - particles[i]))
            
            v_max = 0.1
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] = particles[i] + velocities[i]
            
            # Projection
            dists = np.linalg.norm(particles[i], axis=1)
            mask = dists > 0.99
            if np.any(mask):
                particles[i][mask] *= 0.99 / dists[mask][:, np.newaxis]
            
            new_fitness = _raw_score(particles[i])
            
            if new_fitness > personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = new_fitness
                
                if new_fitness > global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = new_fitness
        
        # Local refinement every 20 iterations on top 3
        if (iteration + 1) % 20 == 0:
            top_indices = np.argsort(personal_best_fitness)[::-1][:3]
            for idx in top_indices:
                refined = refine(personal_best[idx])
                ref_fitness = _raw_score(refined)
                
                if ref_fitness > personal_best_fitness[idx]:
                    personal_best[idx] = refined
                    personal_best_fitness[idx] = ref_fitness
                    particles[idx] = refined.copy()
                    velocities[idx] = 0.0
                    
                    if ref_fitness > global_best_fitness:
                        global_best = refined.copy()
                        global_best_fitness = ref_fitness
                        
            # Reseed stagnant
            stagnant_mask = personal_best_fitness < global_best_fitness * 0.95
            if np.any(stagnant_mask):
                for i in np.where(stagnant_mask)[0]:
                    particles[i] = _init_particle()
                    velocities[i] = 0.0
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = _raw_score(particles[i])
                    
                    if personal_best_fitness[i] > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = personal_best_fitness[i]
        
        if global_best_fitness > 0.1255:
            break
    
    return global_best

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    positions = _construct_pso()
    
    # Final refinement
    def neg_score(x_flat):
        pos = x_flat.reshape(N, 2)
        return -_raw_score(pos)
    
    try:
        result = minimize(neg_score, positions.flatten(), method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        new_pos = result.x.reshape(N, 2)
        if _raw_score(new_pos) > _raw_score(positions):
            positions = new_pos
    except Exception:
        pass
    
    # Final safety check
    dist = np.linalg.norm(positions, axis=1)
    mask = dist > 0.9999
    if np.any(mask):
        positions[mask] *= 0.9999 / dist[mask][:, np.newaxis]
        
    return positions
# EVOLVE-BLOCK-END
