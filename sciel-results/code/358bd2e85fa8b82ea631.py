import numpy as np
import random
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin, angle 0
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Normals to the sides of the outer hexagon
    outer_normals = np.array([
        [math.cos(math.pi/6 + i * math.pi / 3), math.sin(math.pi/6 + i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Precompute min and max projections of unit hexagon onto outer normals
    proj_matrix = hex_verts @ outer_normals.T  # 6x6
    max_proj_inner = np.max(proj_matrix, axis=0)  # (6,)
    min_proj_inner = np.min(proj_matrix, axis=0)  # (6,)
    
    # Helper to compute minimal enclosing hexagon side length for given positions
    def compute_S(pos):
        projs = pos @ outer_normals.T  # Nx6
        projs += max_proj_inner  # Nx6
        max_val = np.max(projs)
        S_req = max_val * 2.0 / math.sqrt(3)
        return S_req
    
    # Helper to check if two hexagons overlap
    def check_overlap(pos_i, pos_j):
        dist_sq = np.sum((pos_i - pos_j)**2)
        if dist_sq > 4.0:
            return 0.0
        
        min_penetration = float('inf')
        
        for k in range(6):
            n = outer_normals[k]
            min_p_i = pos_i @ n + min_proj_inner[k]
            max_p_i = pos_i @ n + max_proj_inner[k]
            min_p_j = pos_j @ n + min_proj_inner[k]
            max_p_j = pos_j @ n + max_proj_inner[k]
            
            if max_p_i < min_p_j or max_p_j < min_p_i:
                return 0.0  # Separated
            
            pen = min(max_p_i, max_p_j) - max(min_p_i, min_p_j)
            if pen < min_penetration:
                min_penetration = pen
        
        return min_penetration
    
    # Helper to compute total overlap penalty
    def compute_penalty(pos):
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                pen = check_overlap(pos[i], pos[j])
                if pen > 0:
                    penalty += pen
        return penalty
    
    # Helper to compute total cost
    def compute_cost(pos):
        S = compute_S(pos)
        penalty = compute_penalty(pos)
        return S + penalty * 100.0

    # Discrete Grid Definition
    # We use a hexagonal grid with spacing h
    h = 0.05
    # Define grid points in a bounding box around the origin
    # The packing should be within S ~ 2.5, so radius ~ 2.5
    # Hex grid coordinates (u, v) -> Cartesian
    # x = h * (u + v/2)
    # y = h * (sqrt(3)/2 * v)
    
    # Generate a set of candidate grid points
    # We need points within a radius of ~3.0
    max_coord = int(3.0 / h) + 2
    grid_points = []
    for u in range(-max_coord, max_coord + 1):
        for v in range(-max_coord, max_coord + 1):
            x = h * (u + v/2.0)
            y = h * (math.sqrt(3)/2.0 * v)
            if x*x + y*y <= 9.0: # Radius 3
                grid_points.append((x, y))
                
    grid_points = np.array(grid_points)
    num_grid_points = len(grid_points)
    
    # Map from position tuple to index for fast lookup? 
    # Or just use indices into grid_points.
    # State is an array of indices into grid_points.
    
    def get_pos(indices):
        return grid_points[indices]

    def get_indices_from_pos(pos):
        # Snap to nearest grid point
        indices = []
        for p in pos:
            # Find nearest neighbor in grid_points
            dists = np.sum((grid_points - p)**2, axis=1)
            idx = np.argmin(dists)
            indices.append(idx)
        return np.array(indices, dtype=int)

    # Initial State: Honeycomb cluster snapped to grid
    def generate_initial_indices():
        positions = []
        # 4x3 rectangular patch of hex grid (standard honeycomb)
        # Side length 1, so distance between centers is sqrt(3) ~ 1.732
        # Wait, for unit hexagons, the distance between centers in a honeycomb is sqrt(3) if they share an edge?
        # No, distance between centers of adjacent unit hexagons is sqrt(3).
        # Let's use the standard lattice vectors.
        # a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
        a1 = np.array([math.sqrt(3), 0])
        a2 = np.array([math.sqrt(3)/2, 1.5])
        
        pts = []
        # 4x3 grid
        for i in range(4):
            for j in range(3):
                p = i * a1 + j * a2
                pts.append(p)
        
        centers = np.array(pts)
        mean_c = np.mean(centers, axis=0)
        centers = centers - mean_c
        
        return get_indices_from_pos(centers)

    # Simulated Annealing on Discrete Grid
    def simulated_annealing_discrete(init_indices, max_time=110.0):
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        start_time = time.time()
        
        current_indices = init_indices.copy()
        current_pos = get_pos(current_indices)
        current_cost = compute_cost(current_pos)
        
        best_indices = current_indices.copy()
        best_cost = current_cost
        best_pos = current_pos.copy()
        
        # Precompute neighbors? No, just pick random grid point within a radius.
        # Or pick a random grid point from a local window.
        
        while T > min_T:
            if time.time() - start_time > max_time:
                break
            
            idx = random.randint(0, N - 1)
            old_grid_idx = current_indices[idx]
            
            # Choose a new grid point
            # Strategy: Pick a random grid point within a certain distance from current
            # To ensure connectivity, we can pick a random point from the grid, 
            # but that might be too large a jump. 
            # Let's pick a random point from a subset of grid points near the current one.
            
            # Simple approach: Pick random index from grid_points.
            # This is a "random restart" within the grid for that particle.
            # Better: Pick a random point within a radius R of current position.
            R = 0.5
            current_p = grid_points[old_grid_idx]
            dists_sq = np.sum((grid_points - current_p)**2, axis=1)
            candidates = np.where(dists_sq < R*R)[0]
            
            if len(candidates) == 0:
                candidates = np.array([old_grid_idx])
            
            new_grid_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            if new_grid_idx == old_grid_idx:
                T *= cooling
                continue
                
            current_indices[idx] = new_grid_idx
            new_pos = get_pos(current_indices)
            new_cost = compute_cost(new_pos)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_indices = current_indices.copy()
                    best_pos = new_pos.copy()
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        current_indices[idx] = old_grid_idx
            
            T *= cooling
            
        return best_pos, best_cost, best_indices

    best_positions = None
    best_S = float('inf')
    
    num_starts = 10
    total_time = 115.0
    time_per_start = total_time / num_starts
    
    for start in range(num_starts):
        if start == 0:
            init_ind = generate_initial_indices()
        else:
            # Perturb initial positions
            init_ind = generate_initial_indices()
            # Randomly swap some indices or shift
            for _ in range(5):
                i = random.randint(0, N-1)
                # Move to a nearby random grid point
                p = grid_points[init_ind[i]]
                dists_sq = np.sum((grid_points - p)**2, axis=1)
                candidates = np.where(dists_sq < 1.0)[0]
                if len(candidates) > 0:
                    init_ind[i] = candidates[random.randint(0, len(candidates) - 1)]
        
        pos, cost, inds = simulated_annealing_discrete(init_ind, max_time=time_per_start)
        
        # Verify no overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if check_overlap(pos[i], pos[j]) > 1e-6:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            S = compute_S(pos)
            if S < best_S:
                best_S = S
                best_positions = pos.copy()
                
    if best_positions is None:
        # Fallback
        init_ind = generate_initial_indices()
        pos, cost, inds = simulated_annealing_discrete(init_ind, max_time=total_time)
        best_positions = pos
        best_S = compute_S(pos)
        
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
