import numpy as np

N = 50

def construct():
    # We will run a local search optimization to maximize the minimum radius.
    # The score is r = min(min_pairwise_dist / 2, min_boundary_dist)
    
    best_global_r = 0.0
    best_global_pos = None
    
    # Number of restarts to avoid local minima
    n_restarts = 10
    n_iterations = 5000
    
    for _ in range(n_restarts):
        # Initialization
        if np.random.rand() > 0.5:
            # Hexagonal initialization
            s = 0.14  # approximate spacing
            # Generate hex grid
            xs = []
            ys = []
            # Range to cover the square [-0.5, 0.5]
            # Hex grid points:
            # x = i*s + (j%2)*s/2
            # y = j*s*sqrt(3)/2
            max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
            max_i = int(1.0 / s) + 2
            for j in range(-max_j, max_j + 1):
                for i in range(-max_i, max_i + 1):
                    x = i * s + (j % 2) * s / 2.0
                    y = j * s * np.sqrt(3) / 2.0
                    # Center the grid roughly
                    # The grid generated is centered around 0,0
                    xs.append(x)
                    ys.append(y)
            
            xs = np.array(xs)
            ys = np.array(ys)
            
            # Filter points inside the square
            mask = (np.abs(xs) <= 0.5) & (np.abs(ys) <= 0.5)
            xs = xs[mask]
            ys = ys[mask]
            
            if len(xs) >= N:
                # Randomly select N points
                indices = np.random.choice(len(xs), N, replace=False)
                pos = np.column_stack((xs[indices], ys[indices]))
            else:
                # If not enough, pad with random points
                pos = np.column_stack((xs, ys))
                remaining = N - len(pos)
                if remaining > 0:
                    extra = np.random.uniform(-0.4, 0.4, (remaining, 2))
                    pos = np.vstack([pos, extra])
        else:
            # Square grid initialization (7x7 + 1)
            m = 7
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            # Take first 49
            pos = grid[:49].copy()
            # Add 50th point at a random location near center or edge
            # Let's add it at a random location
            p50 = np.random.uniform(-0.4, 0.4, 2)
            pos = np.vstack([pos, p50])
            
        # Add small noise to break symmetry
        pos += np.random.normal(0, 0.001, pos.shape)
        
        # Clamp to square
        pos = np.clip(pos, -0.5, 0.5)
        
        # Optimization
        current_r = calculate_radius(pos)
        best_local_r = current_r
        best_local_pos = pos.copy()
        
        step_size = 0.02
        
        for it in range(n_iterations):
            # Pick a random point
            idx = np.random.randint(N)
            old_pos = pos[idx].copy()
            
            # Generate new position
            # Use a random step
            step = np.random.normal(0, step_size, 2)
            new_pos = old_pos + step
            
            # Keep inside square
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            pos[idx] = new_pos
            
            new_r = calculate_radius(pos)
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_pos = pos.copy()
                # Optionally increase step size slightly if stuck? 
                # For now, keep constant or decay slowly
                step_size = max(0.001, step_size * 0.999) 
            else:
                # Revert
                pos[idx] = old_pos
                # Decay step size if we've been failing for a while?
                # Simple decay
                step_size = max(0.001, step_size * 0.999)

        if best_local_r > best_global_r:
            best_global_r = best_local_r
            best_global_pos = best_local_pos
    
    return best_global_pos

def calculate_radius(pos):
    # Calculate min pairwise distance
    # N=50, so 50x50 distance matrix is fine
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Calculate min boundary distance
    # min_i min_k (1/2 - |c_ik|)
    # This is 0.5 - max(|c_ik|)
    max_abs = np.max(np.abs(pos))
    boundary_dist = 0.5 - max_abs
    
    return min(min_pairwise / 2.0, boundary_dist)
