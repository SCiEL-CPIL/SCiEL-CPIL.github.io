import numpy as np
import time

N = 50

def _score(centres):
    """
    Calculate the radius r for the given centres.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    # Check bounds first to be safe, though clipping ensures this
    # If any centre is outside, it's invalid. But we clip, so they are inside.
    
    # Pairwise distances
    # centres is (N, 3)
    # We need min distance between any pair
    # Using broadcasting
    # diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # This creates a 50x50x3 array, which is fine.
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    # Wall distances
    # min_i min_k (0.5 - |c_ik|)
    # c_ik is in [-0.5, 0.5]
    # dist_to_wall_i = min(0.5 - c_i0, 0.5 - c_i1, 0.5 - c_i2, 
    #                       c_i0 - (-0.5), c_i1 - (-0.5), c_i2 - (-0.5))
    # Since c is in [-0.5, 0.5], 0.5 - |c| is the distance to the closest wall face?
    # No.
    # If c = 0.4, dist to right wall (0.5) is 0.1. dist to left wall (-0.5) is 0.9.
    # 0.5 - |0.4| = 0.1. Correct.
    # If c = -0.4, dist to left wall is 0.1.
    # 0.5 - |-0.4| = 0.1. Correct.
    # So min_k (0.5 - |c_ik|) is the distance to the closest wall for sphere i.
    
    abs_centres = np.abs(centres)
    min_wall_dist = np.min(0.5 - abs_centres, axis=1)
    min_global_wall_dist = np.min(min_wall_dist)
    
    # The radius is half the min pair distance, and also limited by the min wall distance.
    # The spheres are disjoint if 2*r <= min_pair_dist.
    # The spheres are inside if r <= min_wall_dist.
    # So r = min(min_pair_dist / 2, min_wall_dist)
    
    return min(min_pair_dist / 2.0, min_global_wall_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Initialization: Simple grid
    # 50 spheres. 4x4x4 = 64. We take 50.
    # Or 5x5x2 = 50.
    # Let's use a 4x4x4 grid and pick the first 50, or center them better.
    # A 5x5x2 grid:
    # x: 5, y: 5, z: 2
    # Spacing: 1/5 = 0.2 in x,y. 1/2 = 0.5 in z? No, 2 layers.
    # If we use 5x5x2, the z-coordinates could be -0.25 and 0.25.
    # x, y coords: -0.4, -0.2, 0.0, 0.2, 0.4.
    
    m_x, m_y, m_z = 5, 5, 2
    x_ticks = (np.arange(m_x) + 0.5) / m_x - 0.5
    y_ticks = (np.arange(m_y) + 0.5) / m_y - 0.5
    z_ticks = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    centres = np.array([(x, y, z) for z in z_ticks for y in y_ticks for x in x_ticks])
    # This gives 5*5*2 = 50 points.
    
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    step_size = 0.05  # Initial step size for noise
    
    while time.time() < deadline:
        # Check if we should stop early if T is too low and no improvement?
        # For now, just run until time limit.
        
        # Generate new state
        noise = np.random.normal(0, step_size, size=(N, 3))
        new_centres = centres + noise
        
        # Clip to cube [-0.5, 0.5]
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_score = _score(new_centres)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            centres = new_centres
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centres = centres.copy()
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally restart or reduce step size
            step_size *= 0.95
            if step_size < 1e-5:
                break

    return best_centres
