"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _force_directed_layout(rng, n_iters=5000, deadline=None):
    """
    Generate initial configuration using force-directed layout.
    Repulsive forces between circles, attractive force toward center.
    """
    # Start with random positions in the square
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    
    # Force-directed parameters
    repulsion_strength = 0.01
    attraction_strength = 0.001
    damping = 0.9
    velocity = np.zeros((N, 2))
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros((N, 2))
        
        # Repulsive forces between all pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Force magnitude: k / d^2 (inverse square law)
        # Only consider forces from nearby circles for efficiency
        mask = dists < 0.3
        mask[np.diag_indices(N)] = False
        
        for i in range(N):
            for j in range(N):
                if mask[i, j]:
                    d = dists[i, j]
                    if d > 1e-8:
                        # Repulsive force
                        f = repulsion_strength / (d * d)
                        forces[i] += f * diff[i, j] / d
        
        # Attractive force toward center
        forces -= attraction_strength * centers
        
        # Update velocities and positions
        velocity = damping * velocity + forces
        centers += velocity
        
        # Keep within bounds
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Force-directed layout for initial configuration
    fd_deadline = start_time + 10.0  # Allow 10s for force-directed
    best_centers = _force_directed_layout(rng, n_iters=5000, deadline=fd_deadline)
    
    # Add small jitter to break any remaining symmetry
    best_centers = best_centers + rng.normal(0, 0.005, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 2: Simulated Annealing refinement
    T_start = 0.03
    T_end = 1e-6
    alpha = 0.9998  # cooling rate per step
    max_steps = 300000  # upper bound, will be limited by time
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate perturbation
        idx = rng.integers(0, N)
        
        # Step size decays over time
        progress = step / max_steps
        step_size = 0.015 * (1.0 - 0.5 * progress)
        
        delta = rng.normal(0, step_size, 2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp((new_radius - current_radius) / T)
            T = max(T_end, T_start * (alpha ** step))
            
            delta_r = new_radius - current_radius
            if delta_r < -10 * T:
                prob = 0.0
            else:
                prob = np.exp(delta_r / T)
                
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
