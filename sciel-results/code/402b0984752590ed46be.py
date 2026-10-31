import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # State: positions (N, 2) and width W. Height H = 2 - W.
    # Initialize with a grid in a 1x1 square (W=1, H=1)
    side = int(np.ceil(np.sqrt(N)))
    # Create grid positions
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    init_pos = np.column_stack((grid_x.ravel()[:N], grid_y.ravel()[:N]))
    
    # Add small random noise to break symmetry
    init_pos += np.random.normal(0, 0.01, init_pos.shape)
    
    # Ensure positions are within (0, 1)
    init_pos = np.clip(init_pos, 0.01, 0.99)
    
    W_init = 1.0
    
    def evaluate(pos, W):
        H = 2.0 - W
        # Check validity: W and H must be positive
        if W <= 0.01 or H <= 0.01:
            return -np.inf
        # Check if positions are within bounds (with small margin)
        # If any position is outside, penalize heavily
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > W) or np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > H):
            # Calculate how far out they are to create a smooth penalty? 
            # Or just return -inf to reject.
            # For SA, -inf is fine.
            return -np.inf
            
        # Compute pairwise distances
        # pos is (N, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq)
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        min_dist_to_other = np.min(dist, axis=1) / 2.0
        
        # Distance to boundaries
        dx = np.minimum(pos[:, 0], W - pos[:, 0])
        dy = np.minimum(pos[:, 1], H - pos[:, 1])
        dist_to_boundary = np.minimum(dx, dy)
        
        radii = np.minimum(dist_to_boundary, min_dist_to_other)
        
        # Sum of radii
        return np.sum(radii)

    # SA Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    pos = init_pos.copy()
    W = W_init
    best_pos = pos.copy()
    best_W = W
    best_score = evaluate(pos, W)
    
    current_score = best_score
    
    # Pre-allocate arrays for speed? Not strictly necessary for N=21.
    
    while time.time() < deadline and T > T_min:
        # Propose move
        # Perturb 1 to 3 circles
        n_perturb = np.random.randint(1, 4)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        # Perturbation size depends on T
        scale = 0.1 * (T / T_init) + 0.001 # Adjust scale
        # Actually, let's use a fixed scale that decays or is constant.
        # Standard SA often uses fixed step size or adaptive.
        # Let's use a step size that scales with T.
        step = 0.05 * (T / 0.05) + 0.005
        
        new_pos = pos.copy()
        new_pos[idx] += np.random.normal(0, step, (n_perturb, 2))
        
        # Perturb W occasionally
        if np.random.random() < 0.2:
            new_W = W + np.random.normal(0, 0.05)
            # Keep W in reasonable range [0.5, 1.5] to avoid extreme aspect ratios
            new_W = np.clip(new_W, 0.5, 1.5)
        else:
            new_W = W
            
        new_score = evaluate(new_pos, new_W)
        
        if new_score == -np.inf:
            # Reject invalid moves
            T *= cooling_rate
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            pos = new_pos
            W = new_W
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_W = W
                
        T *= cooling_rate

    # Final refinement: Local search / Hill climbing on best solution
    # Use a smaller step size to polish
    pos = best_pos.copy()
    W = best_W
    current_score = best_score
    
    T_local = 0.01
    for _ in range(5000): # Fixed number of steps for polish
        if time.time() > deadline - 5: break
        
        n_perturb = np.random.randint(1, 3)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        step = 0.01
        new_pos = pos.copy()
        new_pos[idx] += np.random.normal(0, step, (n_perturb, 2))
        
        if np.random.random() < 0.1:
            new_W = np.clip(W + np.random.normal(0, 0.01), 0.5, 1.5)
        else:
            new_W = W
            
        new_score = evaluate(new_pos, new_W)
        
        if new_score != -np.inf and new_score > current_score:
            pos = new_pos
            W = new_W
            current_score = new_score
            best_score = current_score
            best_pos = pos.copy()
            best_W = W
            
    # Construct final array
    H = 2.0 - best_W
    # Calculate final radii
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    min_dist_to_other = np.min(dist, axis=1) / 2.0
    
    dx = np.minimum(best_pos[:, 0], best_W - best_pos[:, 0])
    dy = np.minimum(best_pos[:, 1], H - best_pos[:, 1])
    dist_to_boundary = np.minimum(dx, dy)
    
    radii = np.minimum(dist_to_boundary, min_dist_to_other)
    
    # Ensure radii are positive and valid
    # Clip to 0 just in case of numerical issues
    radii = np.maximum(radii, 0.0)
    
    result = np.column_stack((best_pos, radii))
    
    return result

# Note: T_init is used in step calculation, need to define it
T_init = 0.05
