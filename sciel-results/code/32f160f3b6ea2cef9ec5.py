import numpy as np

N = 50

def construct():
    # Simulated Annealing approach
    # Objective: Maximize r = min(min_pairwise/2, min_boundary)
    # Energy: -r (we minimize energy)
    
    best_global_r = 0.0
    best_global_pos = None
    
    n_restarts = 20
    n_iterations = 20000
    T_start = 0.05
    T_end = 0.0001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    for _ in range(n_restarts):
        # Initialization: Hexagonal grid
        pos = initialize_hex()
        
        # Add small noise
        pos += np.random.normal(0, 0.005, pos.shape)
        pos = np.clip(pos, -0.5, 0.5)
        
        current_r = calculate_radius(pos)
        best_local_r = current_r
        best_local_pos = pos.copy()
        
        T = T_start
        
        for it in range(n_iterations):
            T *= cooling_rate
            
            # Pick a random point
            idx = np.random.randint(N)
            old_pos = pos[idx].copy()
            
            # Generate new position
            # Step size should be proportional to T
            step_size = 0.02 * (T / T_start) + 0.001
            step = np.random.normal(0, step_size, 2)
            new_pos = old_pos + step
            
            # Keep inside square
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            pos[idx] = new_pos
            
            new_r = calculate_radius(pos)
            
            # Simulated Annealing acceptance
            if new_r > current_r:
                current_r = new_r
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_pos = pos.copy()
            else:
                # Accept worse move with probability exp((new_r - current_r) / T)
                # Note: new_r < current_r, so exponent is negative
                prob = np.exp((new_r - current_r) / T)
                if np.random.rand() < prob:
                    current_r = new_r
                else:
                    # Revert
                    pos[idx] = old_pos
        
        if best_local_r > best_global_r:
            best_global_r = best_local_r
            best_global_pos = best_local_pos
    
    return best_global_pos

def initialize_hex():
    # Create a hexagonal grid of points
    # Spacing s such that we get roughly 50 points in [-0.5, 0.5]^2
    # Area = 1.0. Density = 50.
    # Hex cell area = s^2 * sqrt(3)/2.
    # 50 * s^2 * sqrt(3)/2 = 1 => s^2 = 2 / (50 * sqrt(3)) => s = sqrt(2/(50*1.732)) = sqrt(0.023) = 0.152
    s = 0.15
    
    points = []
    # Range of indices
    # x = i*s + (j%2)*s/2
    # y = j*s*sqrt(3)/2
    # We need to cover [-0.5, 0.5]
    max_j = int(0.5 / (s * np.sqrt(3) / 2)) + 2
    max_i = int(0.5 / s) + 2
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            # Centering adjustment: The grid is centered at 0,0.
            # We just collect points inside the square.
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        # Select N points. To be deterministic-ish but varied, we can shuffle and take first N
        # Or select the ones closest to a regular grid pattern?
        # Let's just shuffle and take N.
        np.random.shuffle(points)
        pos = points[:N].copy()
    else:
        # If not enough, pad with random
        pos = points.copy()
        remaining = N - len(pos)
        if remaining > 0:
            extra = np.random.uniform(-0.4, 0.4, (remaining, 2))
            pos = np.vstack([pos, extra])
            
    return pos

def calculate_radius(pos):
    # Calculate min pairwise distance
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Calculate min boundary distance
    max_abs = np.max(np.abs(pos))
    boundary_dist = 0.5 - max_abs
    
    return min(min_pairwise / 2.0, boundary_dist)
