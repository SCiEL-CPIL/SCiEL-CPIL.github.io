import numpy as np
import time

N = 50

def calc_radius(centers):
    """
    Calculate the maximum radius for a set of centers in [-0.5, 0.5]^3.
    Returns the radius r.
    """
    # Check validity
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Boundary distance
    # min_i min_k (0.5 - |c_ik|)
    abs_centers = np.abs(centers)
    boundary_dists = 0.5 - np.max(abs_centers, axis=1)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
        
    # Pairwise distances
    # Compute distance matrix
    # diff[i, j] = c_i - c_j
    # dist[i, j] = norm(diff[i, j])
    # We only need i < j
    # Vectorized approach:
    # c = centers
    # d = sqrt(sum((c[:, None, :] - c[None, :, :])**2, axis=-1))
    # This creates a N x N matrix.
    
    c = centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Mask the diagonal and upper triangle to get unique pairs
    # Or just take min of all off-diagonal elements
    # np.fill_diagonal(dists, np.inf)
    # min_pair = np.min(dists)
    
    # More efficient:
    # Get indices of upper triangle
    i, j = np.triu_indices(N, k=1)
    min_pair = np.min(dists[i, j])
    
    r = min(min_boundary, min_pair / 2.0)
    return r

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Initial guess: Jittered grid
    # 4x4x4 grid is 64 points, we need 50.
    # Let's use a 4x4x4 grid and take the first 50, then jitter.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Ensure inside cube
    centers = np.clip(centers, -0.49, 0.49)
    
    current_r = calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Simulated Annealing
    T = 0.01
    T_min = 0.0001
    cooling_rate = 0.999
    
    # We will run steps until time limit
    # Estimate steps: calc_radius is ~0.1ms? Let's check.
    # N=50, dist matrix 50x50. Very fast.
    # Let's say 1000 steps per second is conservative.
    
    step_count = 0
    while time.time() - start_time < time_limit:
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random move
        # Step size proportional to T?
        # Let's use a fixed step size or scaled by T
        step_size = 0.05 * T / 0.01 # Scale with T, max step 0.05
        move = np.random.uniform(-step_size, step_size, 3)
        
        new_centers = centers.copy()
        new_centers[idx] += move
        
        # Check bounds
        if np.any(new_centers[idx] < -0.5) or np.any(new_centers[idx] > 0.5):
            # Reflect or reject? Reject is safer for validity, but reflection might explore better.
            # Let's clip to stay inside, or reject.
            # If we clip, the move is smaller.
            # Let's just reject if out of bounds to keep it simple and valid.
            continue
        
        new_r = calc_radius(new_centers)
        
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            if delta_r < -T * 10: # Optimization: skip if too bad
                pass
            else:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    centers = new_centers
                    current_r = new_r
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Maybe restart with higher T if stuck?
            # For now, just continue with low T for polish.
            
        step_count += 1
        # If we are running very fast, we can batch steps?
        # No, SA is sequential.
        
    return best_centers
