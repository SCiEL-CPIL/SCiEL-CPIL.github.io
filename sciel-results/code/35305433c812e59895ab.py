import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_config = None
    
    # Helper to check validity and score
    def check_config(config):
        # config: (N, 3) array of [x, y, r]
        x = config[:, 0]
        y = config[:, 1]
        r = config[:, 2]
        
        # Check boundaries
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
            
        # Check overlaps
        # d_ij >= r_i + r_j
        # Use broadcasting for pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        sum_radii = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Diagonal is 0, so we ignore it
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(sum_radii, 0)
        
        # If any dist < sum_radii, it's invalid
        if np.any(dist < sum_radii - 1e-9):
            return False, 0.0
            
        return True, np.sum(r)

    def run_optimization(init_config, max_iter=2000):
        # init_config: (N, 3)
        config = init_config.copy()
        x = config[:, 0]
        y = config[:, 1]
        r = config[:, 2]
        
        # We will use a simple gradient descent on positions to resolve overlaps,
        # and then try to grow radii.
        
        # Actually, a better heuristic:
        # 1. Relax positions to remove overlaps for current radii.
        # 2. If successful, increase radii.
        
        current_r = r.copy()
        pos = np.column_stack((x, y))
        
        step_size = 0.01
        growth_factor = 1.001
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # 1. Relaxation step
            # Calculate forces
            forces = np.zeros_like(pos)
            
            # Pairwise forces
            dx = pos[:, 0][:, np.newaxis] - pos[:, 0][np.newaxis, :]
            dy = pos[:, 1][:, np.newaxis] - pos[:, 1][np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, 1.0) # Avoid div by zero
            
            # Overlap amount
            sum_r = current_r[:, np.newaxis] + current_r[np.newaxis, :]
            overlap = np.maximum(sum_r - dist, 0)
            
            # Force direction
            # If dist is 0, pick random direction
            for i in range(N):
                for j in range(i+1, N):
                    if overlap[i, j] > 0:
                        # Vector from i to j
                        dir_x = pos[j, 0] - pos[i, 0]
                        dir_y = pos[j, 1] - pos[i, 1]
                        d = np.sqrt(dir_x**2 + dir_y**2)
                        if d < 1e-9:
                            dir_x, dir_y = np.random.randn(2)
                            d = np.sqrt(dir_x**2 + dir_y**2)
                        
                        # Push apart
                        # Force proportional to overlap
                        f = overlap[i, j]
                        forces[i, 0] -= dir_x / d * f
                        forces[i, 1] -= dir_y / d * f
                        forces[j, 0] += dir_x / d * f
                        forces[j, 1] += dir_y / d * f
            
            # Wall forces
            # Push in if outside
            for i in range(N):
                if pos[i, 0] < current_r[i]:
                    forces[i, 0] += (current_r[i] - pos[i, 0]) * 10
                if pos[i, 0] > 1 - current_r[i]:
                    forces[i, 0] -= (pos[i, 0] - (1 - current_r[i])) * 10
                if pos[i, 1] < current_r[i]:
                    forces[i, 1] += (current_r[i] - pos[i, 1]) * 10
                if pos[i, 1] > 1 - current_r[i]:
                    forces[i, 1] -= (pos[i, 1] - (1 - current_r[i])) * 10
            
            # Update positions
            pos += forces * step_size
            
            # Clamp positions to valid range roughly
            # This helps stability
            pos[:, 0] = np.clip(pos[:, 0], current_r, 1 - current_r)
            pos[:, 1] = np.clip(pos[:, 1], current_r, 1 - current_r)
            
            # Check if valid
            test_config = np.column_stack((pos, current_r))
            is_valid, score = check_config(test_config)
            
            if is_valid:
                # Try to grow
                new_r = current_r * growth_factor
                # Check if growth is possible (just check boundaries for now, overlaps will be fixed in next iter)
                if np.all(new_r <= 0.5): # Max possible radius is 0.5
                    current_r = new_r
                else:
                    # Reduce step size if we can't grow
                    step_size *= 0.9
            else:
                # If invalid, reduce step size to converge
                step_size *= 0.95
                if step_size < 1e-5:
                    break
            
            config = np.column_stack((pos, current_r))
        
        # Final check
        is_valid, score = check_config(config)
        if is_valid:
            return config, score
        else:
            return None, 0.0

    # Initializations
    inits = []
    
    # 1. Hexagonal-like packing
    side = int(np.ceil(np.sqrt(N)))
    hex_points = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            if i % 2 == 1:
                y += 0.25 / side
            hex_points.append([x, y, 0.1])
    inits.append(np.array(hex_points[:N]))
    
    # 2. Grid
    grid_points = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            grid_points.append([x, y, 0.1])
    inits.append(np.array(grid_points[:N]))
    
    # 3. Random
    for _ in range(5):
        rand_points = np.random.rand(N, 2)
        rand_r = np.full(N, 0.05)
        inits.append(np.column_stack((rand_points, rand_r)))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Run optimization
        config, score = run_optimization(init)
        
        if config is not None and score > best_score:
            best_score = score
            best_config = config
            
    if best_config is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
