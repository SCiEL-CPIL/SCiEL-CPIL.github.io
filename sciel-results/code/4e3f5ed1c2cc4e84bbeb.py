import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # Initial random configuration
    # We want to start with a configuration that is valid and reasonably spread out.
    # A simple random uniform distribution might have overlaps, which is fine for MC, 
    # but we can try to start with a jittered grid or just random.
    # Let's use random uniform in [-0.45, 0.45] to ensure they start inside.
    np.random.seed(42) # Fixed seed for reproducibility in evaluation if needed, but random is better for exploration?
    # Actually, better to use a fixed seed for consistent benchmarking or just default.
    # Let's use a default random seed to allow different runs to potentially find different local optima.
    # However, for a deterministic test, fixed seed is safer. Let's stick to random but ensure validity.
    
    # Start with a grid-like initial guess to avoid immediate deep local minima of overlapping spheres
    # m = 4 -> 64 points, take 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add some random jitter to break symmetry
    centers = grid[:N] + np.random.normal(0, 0.05, (N, 3))
    # Clip to ensure they are inside [-0.5, 0.5]
    centers = np.clip(centers, -0.49, 0.49)
    
    def calc_radius(centers):
        # Calculate minimum distance to walls
        wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
        r_wall = np.min(wall_dist)
        
        # Calculate minimum pair distance
        # Use broadcasting for pairwise distances
        # dists[i, j] = |c_i - c_j|
        # To save memory and time, we can compute the full matrix for N=50 (50x50 is small)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Exclude diagonal (distance 0)
        np.fill_diagonal(dists, np.inf)
        
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)

    current_r = calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # MC Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.99995
    step_size = 0.05
    
    # Pre-compute indices for pair distances? No, N is small enough to just recompute.
    
    while time.time() - start_time < timeout:
        # Pick a random sphere
        idx = np.random.randint(N)
        
        # Generate a random displacement
        # Use Gaussian steps for smoother exploration
        delta = np.random.normal(0, step_size, 3)
        
        new_center = centers[idx].copy()
        new_center += delta
        
        # Check boundaries
        if np.any(new_center < -0.5) or np.any(new_center > 0.5):
            # Reject if outside
            continue
            
        # Update center
        centers[idx] = new_center
        
        # Calculate new radius
        new_r = calc_radius(centers)
        
        # Metropolis Criterion
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Note: new_r - current_r is negative
            delta_r = new_r - current_r
            if delta_r > -T * 10: # Optimization: avoid underflow/overflow in exp
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_r = new_r
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = centers.copy()
            # else reject
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally reheat or increase step size if stuck?
        # For now, keep it simple.
        
    # Final validation and return
    # Ensure the returned centers are valid
    if np.any(best_centers < -0.5) or np.any(best_centers > 0.5):
        # Fallback to a safe grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
