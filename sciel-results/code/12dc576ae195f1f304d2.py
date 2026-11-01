import numpy as np
import time

N = 100

def _calc_radius(centers):
    """
    Calculate the maximum radius for the given centers.
    centers: (N, 3) array
    """
    # Distance to walls
    # Centers must be in [-0.5, 0.5]
    # dist to wall = min(0.5 - x, x + 0.5) for each coord
    # Equivalent to 0.5 - max(abs(x), abs(y), abs(z)) for each sphere?
    # No, min_k (0.5 - |c_ik|).
    # min over k of (0.5 - |c_ik|) is 0.5 - max_k |c_ik|.
    # So wall_dist_i = 0.5 - np.max(np.abs(centers[i]), axis=1)
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # This creates a 100x100x3 array.
    # To save memory/time, we can do it in chunks or just accept the memory for N=100.
    # 100*100*3*8 bytes = 240KB. Negligible.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Radius is min(min_wall_dist, min_pair_dist / 2)
    return min(min_wall_dist, min_pair_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    current_r = _calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Simulated Annealing Parameters
    T = 0.01 # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.999
    
    # Step size scaling
    step_scale = 0.1
    
    while time.time() < deadline:
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Generate random displacement
        # Use Gaussian for smoother exploration
        delta = np.random.normal(0, step_scale, 3)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Keep within bounds
        # If it goes out, clip it? Or reject?
        # Clipping is safer to maintain validity, but might stick to walls.
        # Let's clip.
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _calc_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Probability of accepting worse state
            # Delta E = E_new - E_old = -new_r - (-current_r) = current_r - new_r
            # P = exp(-Delta E / T) = exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    centers = new_centers
                    current_r = new_r
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally restart or increase step size if stuck?
        # Simple strategy: if no improvement in last X steps, increase step size slightly?
        # Or just let it run. The cooling schedule is important.
        
        # Adaptive step size?
        # If we are accepting many moves, maybe step is too big?
        # If we are accepting few, maybe too small?
        # For now, fixed step with cooling T.
        
    return best_centers

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100

def _calc_radius(centers):
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    current_r = _calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    T = 0.005 
    T_min = 1e-6
    cooling_rate = 0.9995
    step_scale = 0.05
    
    # Add some randomization to initial state to break symmetry
    # But keep it close to grid to not lose the good start
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)
    current_r = _calc_radius(centers)
    if current_r > best_r:
        best_r = current_r
        best_centers = centers.copy()
    else:
        # If randomization made it worse, revert to grid
        centers = grid[:N].copy()
        current_r = _calc_radius(centers)

    while time.time() < deadline:
        i = np.random.randint(N)
        delta = np.random.normal(0, step_scale, 3)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _calc_radius(new_centers)
        
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    centers = new_centers
                    current_r = new_r
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_centers
# EVOLVE-BLOCK-END
