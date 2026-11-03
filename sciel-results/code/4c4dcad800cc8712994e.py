import numpy as np
import time
import math
import random

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Fast Simulated Annealing with Spatial Hashing to maximize the minimum radius
    of 200 circles in a unit square.
    """
    
    # Grid parameters for spatial hashing
    GRID_DIM = 20
    CELL_SIZE = 1.0 / GRID_DIM
    
    def to_grid_idx(x, y):
        gx = int((x + 0.5) / CELL_SIZE)
        gy = int((y + 0.5) / CELL_SIZE)
        if gx >= GRID_DIM: gx = GRID_DIM - 1
        if gy >= GRID_DIM: gy = GRID_DIM - 1
        if gx < 0: gx = 0
        if gy < 0: gy = 0
        return gx, gy

    grid = [[[] for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]

    def rebuild_grid(points):
        """Rebuild the spatial grid from scratch."""
        for i in range(GRID_DIM):
            for j in range(GRID_DIM):
                grid[i][j] = []
        for i in range(N):
            gx, gy = to_grid_idx(points[i, 0], points[i, 1])
            grid[gy][gx].append(i)

    def update_grid(points, idx, old_pos, new_pos):
        """Update the spatial grid when point idx moves from old_pos to new_pos."""
        ogx, ogy = to_grid_idx(old_pos[0], old_pos[1])
        ngx, ngy = to_grid_idx(new_pos[0], new_pos[1])
        
        # Remove from old cell
        old_list = grid[ogy][ogx]
        if idx in old_list:
            old_list.remove(idx)
        
        # Add to new cell
        grid[ngy][ngx].append(idx)

    def get_neighbors(grid, gx, gy):
        """Yield indices of points in the 3x3 neighborhood of (gx, gy)."""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                    yield from grid[ny][nx]

    def calculate_min_radius(points):
        """
        Calculate the minimum radius for the entire configuration.
        """
        wx = np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0])
        wy = np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
        wall_dists = np.minimum(wx, wy)
        min_wall = np.min(wall_dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_wall, min_pair / 2.0)

    def evaluate_move(points, idx, new_pos, current_min_r):
        """
        Efficiently evaluate the new minimum radius if point idx moves to new_pos.
        """
        new_wall = min(0.5 - new_pos[0], 0.5 + new_pos[0], 0.5 - new_pos[1], 0.5 + new_pos[1])
        
        gx, gy = to_grid_idx(new_pos[0], new_pos[1])
        
        min_new_pair = float('inf')
        for j in get_neighbors(grid, gx, gy):
            if j == idx:
                continue
            dx = new_pos[0] - points[j, 0]
            dy = new_pos[1] - points[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_new_pair:
                min_new_pair = d
                
        new_pair_r = min_new_pair / 2.0
        new_r = min(current_min_r, new_wall, new_pair_r)
        
        return new_r

    # Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    init_grid = np.array([(x, y) for y in ticks for x in ticks])
    
    best_points = init_grid[:N].copy()
    best_score = calculate_min_radius(best_points)
    
    start_time = time.time()
    time_limit = 110
    
    # Simulated Annealing Parameters
    T0 = 0.01
    T_min = 1e-7
    cooling_rate = 0.999
    step_scale = 0.05
    
    points = best_points.copy()
    current_score = best_score
    T = T0
    
    rebuild_grid(points)
    
    restarts = 0
    max_restarts = 10
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        restarts += 1
        
        if restarts == 1:
            points = best_points.copy()
            points += np.random.normal(0, 0.01, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            rebuild_grid(points)
            current_score = calculate_min_radius(points)
            T = T0
        else:
            points = best_points.copy()
            points += np.random.normal(0, 0.05, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            rebuild_grid(points)
            current_score = calculate_min_radius(points)
            T = T0 * 0.5
            
        iter_count = 0
        while time.time() - start_time < time_limit:
            idx = np.random.randint(N)
            
            move_size = step_scale * T / T0
            if move_size < 1e-5:
                move_size = 1e-5
                
            old_pos = points[idx].copy()
            new_pos = old_pos + np.random.normal(0, move_size, size=2)
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            new_score = evaluate_move(points, idx, new_pos, current_score)
            
            if new_score > current_score:
                update_grid(points, idx, old_pos, new_pos)
                points[idx] = new_pos
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                delta = current_score - new_score
                if delta < 1e-12:
                    accept = True
                else:
                    ratio = delta / T
                    if ratio > 100:
                        prob = 0
                    else:
                        prob = math.exp(-ratio)
                    accept = (np.random.random() < prob)
                
                if accept:
                    update_grid(points, idx, old_pos, new_pos)
                    points[idx] = new_pos
                    current_score = new_score
            
            # Periodic full recalculation to correct conservative estimates
            iter_count += 1
            if iter_count % 100 == 0:
                actual_score = calculate_min_radius(points)
                if actual_score > current_score:
                    current_score = actual_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
            
            T *= cooling_rate
            if T < T_min:
                break
                
    return best_points
# EVOLVE-BLOCK-END
