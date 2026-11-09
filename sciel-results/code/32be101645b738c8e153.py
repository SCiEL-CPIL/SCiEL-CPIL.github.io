import numpy as np
import random
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Precompute normals for outer hexagon
    normals = np.array([
        [math.cos(math.pi/6 + i * math.pi / 3), math.sin(math.pi/6 + i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Helper to compute minimal enclosing hexagon side length for given positions
    def compute_S(pos):
        # pos is Nx2 array
        # S * sqrt(3)/2 >= max_k ( max_i ( pos_i @ n_k ) + 1 )
        projs = pos @ normals.T
        max_projs = np.max(projs, axis=0)
        max_val = np.max(max_projs)
        S = (max_val + 1.0) * 2.0 / math.sqrt(3)
        return S

    # Hexagonal grid parameters
    grid_step = 0.05
    R = 5.0
    num_points = int(R / grid_step)
    
    # Generate grid points
    grid_points = []
    for i in range(-num_points, num_points + 1):
        for j in range(-num_points, num_points + 1):
            x = i * grid_step
            y = j * grid_step
            if x*x + y*y <= R*R:
                grid_points.append((x, y))
                
    grid_points = np.array(grid_points)
    num_grid = len(grid_points)
    
    # Min distance squared
    min_dist_sq = 3.0 # (sqrt(3))^2

    # Local Search on the discrete grid
    def local_search(initial_indices, max_time=110.0):
        current_indices = initial_indices.copy()
        current_pos = grid_points[current_indices]
        current_S = compute_S(current_pos)
        
        best_indices = current_indices.copy()
        best_S = current_S
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        start_time = time.time()
        
        while T > min_T:
            if time.time() - start_time > max_time:
                break
            
            # Pick a random hexagon to move
            idx = random.randint(0, N - 1)
            old_idx = current_indices[idx]
            
            # Pick a new random grid point
            new_idx = random.randint(0, num_grid - 1)
            
            if new_idx == old_idx:
                continue
                
            current_indices[idx] = new_idx
            new_pos = grid_points[current_indices]
            
            # Check validity
            valid = True
            for i in range(N):
                if i == idx: continue
                dist_sq = np.sum((new_pos[idx] - new_pos[i])**2)
                if dist_sq < min_dist_sq - 1e-9:
                    valid = False
                    break
            
            if not valid:
                # Revert
                current_indices[idx] = old_idx
                continue
                
            new_S = compute_S(new_pos)
            
            if new_S < current_S:
                current_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_indices = current_indices.copy()
            else:
                delta = new_S - current_S
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_S = new_S
                else:
                    current_indices[idx] = old_idx
                    
            T *= cooling
            
        return best_indices, best_S

    # Multi-start
    best_S_global = float('inf')
    best_indices_global = None
    
    # Initial configuration: Greedy
    def get_initial_config():
        dists = np.sum(grid_points**2, axis=1)
        sorted_indices = np.argsort(dists)
        
        config = []
        for idx in sorted_indices:
            if len(config) >= N:
                break
            valid = True
            for existing_idx in config:
                dist_sq = np.sum((grid_points[idx] - grid_points[existing_idx])**2)
                if dist_sq < min_dist_sq - 1e-9:
                    valid = False
                    break
            if valid:
                config.append(idx)
                
        if len(config) < N:
            config = [random.randint(0, num_grid - 1) for _ in range(N)]
            
        return np.array(config)

    num_starts = 20
    for s in range(num_starts):
        init_indices = get_initial_config()
        
        if s > 0:
            for _ in range(5):
                idx_to_change = random.randint(0, N - 1)
                new_idx = random.randint(0, num_grid - 1)
                valid = True
                for i in range(N):
                    if i == idx_to_change: continue
                    dist_sq = np.sum((grid_points[new_idx] - grid_points[init_indices[i]])**2)
                    if dist_sq < min_dist_sq - 1e-9:
                        valid = False
                        break
                if valid:
                    init_indices[idx_to_change] = new_idx
        
        indices, S_val = local_search(init_indices, max_time=5.0)
        
        if S_val < best_S_global:
            best_S_global = S_val
            best_indices_global = indices
            
        if best_S_global < 3.5:
            break

    if best_indices_global is None:
        best_indices_global = np.array([0]*N)
        best_S_global = compute_S(grid_points[best_indices_global])
        
    final_pos = grid_points[best_indices_global]
    
    inner = []
    for p in final_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_S_global), 0.0

# EVOLVE-BLOCK-END
