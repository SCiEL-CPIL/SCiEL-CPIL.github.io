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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Simulated Annealing
    # Objective: maximize min(pair_dist/2, wall_dist)
    # We will minimize the negative of this, or just track the radius directly.
    
    T_start = 0.05
    T_end = 1e-6
    alpha = 0.9999 # cooling rate per step
    max_steps = 200000 # upper bound, will be limited by time
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    # Pre-compute indices for efficiency? 
    # For N=50, 1225 pairs. 
    # Vectorized radius computation is fast enough.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate perturbation
        # Random move for one or a few circles
        # Let's pick a random circle and move it by a Gaussian amount
        idx = rng.integers(0, N)
        # Move size scales with temperature? Or fixed?
        # Let's use a move size that decays slightly or is constant.
        # A common choice is to scale with current radius or a fixed small value.
        # Let's try fixed small value initially, maybe 0.01 * T/T_start? No, T is abstract.
        # Let's just use a fixed step size that decreases over time.
        step_size = 0.05 * (T_start / (T_start + step * 0.001)) 
        # Or simpler:
        step_size = 0.02 * (1.0 + 0.5 * np.cos(np.pi * step / max_steps)) # 0.02 to 0.01
        
        delta = rng.normal(0, step_size, 2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        
        # Check bounds
        # If out of bounds, reflect or reject? 
        # Reflecting might be better to stay in feasible space.
        # But we want to maximize radius, so being close to wall is bad unless necessary.
        # Let's just clip to [-0.5, 0.5] and let the radius function penalize it.
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
            # Note: new_radius <= current_radius, so exponent is negative.
            # If T is small, this probability is tiny.
            if T_end < T_start * (alpha ** step) < T_start:
                T = T_start * (alpha ** step)
            else:
                T = T_end
                
            delta_r = new_radius - current_radius
            # Avoid log(0) or overflow
            if delta_r < -10 * T:
                prob = 0.0
            else:
                prob = np.exp(delta_r / T)
                
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
