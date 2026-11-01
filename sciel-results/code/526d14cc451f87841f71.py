"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _min_radius(centers):
    """
    Compute the maximum possible radius for a set of centers.
    r = 0.5 * min(min_pairwise_dist, min_dist_to_boundary)
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to boundaries
    dist_to_boundary = 0.5 - np.abs(centers)
    min_bound = np.min(dist_to_boundary)
    
    return 0.5 * min(min_pair, min_bound)

def _local_search(centers, time_limit=5.0):
    """
    Hill climbing local search.
    Tries to move each center in random directions to increase the minimum radius.
    """
    centers = centers.copy()
    current_r = _min_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    start_time = time.time()
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each center
        for i in range(N):
            # Try several random directions
            for _ in range(10):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                
                new_center = centers[i].copy()
                new_center += step_size * direction
                
                # Check boundaries
                if np.any(np.abs(new_center) > 0.5):
                    continue
                    
                old_center = centers[i].copy()
                centers[i] = new_center
                new_r = _min_radius(centers)
                
                if new_r > current_r:
                    current_r = new_r
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
                else:
                    centers[i] = old_center
                    
        if not improved:
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return best_centers

def _simulated_annealing(centers, time_limit=30.0):
    """
    Simulated Annealing to find a good configuration.
    """
    centers = centers.copy()
    current_r = _min_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.05 # Initial temperature
    cooling = 0.99
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > 1e-6:
        # Pick a random center and move it
        i = np.random.randint(N)
        move_size = 0.05
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        old_pos = centers[i].copy()
        centers[i] += move_size * direction
        
        # Keep inside cube
        centers[i] = np.clip(centers[i], -0.49, 0.49)
        
        new_r = _min_radius(centers)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            centers[i] = old_pos
            
        T *= cooling
        
    return best_centers

def _lattice_init(seed):
    """
    Initialize with a perturbed lattice.
    """
    rng = np.random.RandomState(seed)
    
    # Use a simple cubic lattice scaled to fit
    # 4x4x4 = 64 points, take 50
    # Or 3x3x3 = 27, too few.
    # Let's use a grid that covers the space well.
    
    # Generate a 4x4x4 grid
    coords = np.array(np.meshgrid(
        np.linspace(-0.4, 0.4, 4),
        np.linspace(-0.4, 0.4, 4),
        np.linspace(-0.4, 0.4, 4)
    )).T.reshape(-1, 3)
    
    # Shuffle and take 50
    rng.shuffle(coords)
    centers = coords[:50]
    
    # Add noise
    centers += rng.randn(50, 3) * 0.05
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _random_init(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_centers = None
    best_r = 0.0
    
    # Try a few different initializations
    inits = []
    for seed in range(5):
        inits.append(_lattice_init(seed))
    for seed in range(5):
        inits.append(_random_init(seed))
        
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Phase 1: Simulated Annealing
        remaining = time_limit - (time.time() - start_time)
        sa_time = min(remaining * 0.7, 20.0)
        if sa_time < 1.0:
            break
            
        sa_result = _simulated_annealing(init, time_limit=sa_time)
        
        # Phase 2: Local Search
        remaining = time_limit - (time.time() - start_time)
        ls_time = min(remaining * 0.3, 10.0)
        if ls_time < 0.5:
            break
            
        ls_result = _local_search(sa_result, time_limit=ls_time)
        
        r = _min_radius(ls_result)
        if r > best_r:
            best_r = r
            best_centers = ls_result.copy()
            
    # Final polish on the best result found
    if best_centers is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 2.0:
            final_result = _local_search(best_centers, time_limit=remaining * 0.8)
            final_r = _min_radius(final_result)
            if final_r >= best_r:
                best_centers = final_result
                
    return best_centers

# EVOLVE-BLOCK-END
