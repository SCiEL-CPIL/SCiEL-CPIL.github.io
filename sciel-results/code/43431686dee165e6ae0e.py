"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def construct():
    """
    Pack 21 circles in a rectangle with perimeter 4 (w+h=2).
    Uses Hungarian algorithm to match circles to grid cells,
    then locally optimizes positions to maximize minimum radius.
    """
    best_result = None
    best_sum = -1
    
    # Try different aspect ratios (w + h = 2)
    aspect_ratios = [
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
    ]
    
    # Grid resolutions to try (nx, ny)
    grid_configs = [
        (5, 5), (5, 6), (6, 5), (6, 6), (4, 7), (7, 4), (7, 7),
        (3, 8), (8, 3), (8, 8), (4, 6), (6, 4), (5, 7), (7, 5)
    ]
    
    for w_frac in aspect_ratios:
        w = 2 * w_frac
        h = 2 - w
        
        if w <= 0 or h <= 0:
            continue
        
        for nx, ny in grid_configs:
            if nx * ny < N:
                continue
            
            # Create high-resolution grid of candidate positions
            # Use finer grid for better placement
            grid_nx = nx * 3  # 3x refinement
            grid_ny = ny * 3
            
            # Generate grid positions
            xs = np.linspace(w / (2 * grid_nx), w - w / (2 * grid_nx), grid_nx)
            ys = np.linspace(h / (2 * grid_ny), h - h / (2 * grid_ny), grid_ny)
            
            all_positions = []
            for i in range(grid_nx):
                for j in range(grid_ny):
                    all_positions.append([xs[i], ys[j]])
            all_positions = np.array(all_positions)
            
            n_cells = len(all_positions)
            
            # Build cost matrix: for each circle-cell pair, cost = negative of feasible radius
            # Feasible radius at position (x,y) = min(x, w-x, y, h-y)
            # But we also need to consider spacing to other circles
            
            # First pass: use boundary constraint only
            # Then refine with pairwise distances
            
            # Cost matrix: circle i to cell j
            # We want to maximize minimum radius, so minimize -min_radius
            cost_matrix = np.zeros((N, n_cells))
            
            for i in range(N):
                for j in range(n_cells):
                    x, y = all_positions[j]
                    # Boundary constraint
                    r_boundary = min(x, w - x, y, h - y)
                    # Use negative so that linear_sum_assignment minimizes
                    cost_matrix[i, j] = -r_boundary
            
            # Find optimal assignment using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Get assigned positions
            positions = all_positions[col_ind].copy()
            
            # Now locally optimize: try to increase minimum radius
            # by adjusting positions
            
            # Compute current minimum radius
            def compute_min_radius(pos, w, h):
                min_r = np.inf
                # Pairwise distances
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        min_r = min(min_r, d / 2)
                # Boundary distances
                for i in range(N):
                    d_b = min(pos[i][0], w - pos[i][0],
                             pos[i][1], h - pos[i][1])
                    min_r = min(min_r, d_b)
                return min_r
            
            current_r = compute_min_radius(positions, w, h)
            
            # Local optimization: gradient-free search
            # Try small perturbations in each direction
            step_size = 0.01
            improved = True
            max_iterations = 100
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                
                for i in range(N):
                    for dx, dy in [(step_size, 0), (-step_size, 0), 
                                   (0, step_size), (0, -step_size),
                                   (step_size*0.707, step_size*0.707),
                                   (-step_size*0.707, step_size*0.707),
                                   (step_size*0.707, -step_size*0.707),
                                   (-step_size*0.707, -step_size*0.707)]:
                        # Try moving circle i
                        old_x, old_y = positions[i]
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        # Check bounds
                        if new_x < 0 or new_x > w or new_y < 0 or new_y > h:
                            continue
                        
                        positions[i] = [new_x, new_y]
                        new_r = compute_min_radius(positions, w, h)
                        
                        if new_r > current_r + 1e-10:
                            current_r = new_r
                            improved = True
                        else:
                            positions[i] = [old_x, old_y]
                
                # Decrease step size if no improvement
                if not improved:
                    step_size *= 0.8
                    if step_size < 0.001:
                        break
            
            # Final check
            final_r = compute_min_radius(positions, w, h)
            if final_r > 0:
                total_sum = N * final_r
                if total_sum > best_sum:
                    best_sum = total_sum
                    result = np.zeros((N, 3))
                    result[:, 0] = positions[:, 0]
                    result[:, 1] = positions[:, 1]
                    result[:, 2] = final_r
                    best_result = result
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
