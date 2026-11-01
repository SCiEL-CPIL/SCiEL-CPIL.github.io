"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Grid of candidate centers in a 1x1 square (perimeter 4)
    # We'll use a grid resolution that balances quality and speed
    grid_res = 20
    x_coords = np.linspace(0.05, 0.95, grid_res)
    y_coords = np.linspace(0.05, 0.95, grid_res)
    
    # Generate all candidate positions
    candidates = []
    for x in x_coords:
        for y in y_coords:
            candidates.append((x, y))
    
    # Initial greedy placement to get a starting configuration
    # Sort candidates by some heuristic (e.g., distance from center)
    center_x, center_y = 0.5, 0.5
    candidates_sorted = sorted(candidates, key=lambda p: (p[0]-center_x)**2 + (p[1]-center_y)**2)
    
    # We'll use a local search approach
    # Start with a reasonable initial configuration
    # Place circles in a grid-like pattern initially
    
    # Initial grid placement
    side = 5  # 5x5 grid = 25 cells, use 21
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    
    cells = np.array(cells[:N])
    
    # Local search: try to move circles to increase total radius
    # Use a simple hill-climbing approach
    
    best_sum = np.sum(cells[:, 2])
    best_cells = cells.copy()
    
    # Try to optimize by adjusting positions and radii
    # For each circle, try small perturbations
    
    num_iterations = 0
    max_iterations = 50000
    step_size = 0.05
    
    while num_iterations < max_iterations:
        if time.time() > deadline - 10:
            break
        
        num_iterations += 1
        improved = False
        
        for idx in range(N):
            if time.time() > deadline - 10:
                break
            
            x, y, r = cells[idx]
            
            # Try small perturbations in x, y, r
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    for dr in [-step_size, 0, step_size]:
                        new_x = x + dx
                        new_y = y + dy
                        new_r = max(0.01, r + dr)
                        
                        # Check bounds
                        if new_x < new_r or new_x > 1 - new_r:
                            continue
                        if new_y < new_r or new_y > 1 - new_r:
                            continue
                        
                        # Check overlap with other circles
                        valid = True
                        for jdx in range(N):
                            if jdx == idx:
                                continue
                            ox, oy, orr = cells[jdx]
                            dist = math.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                            if dist < new_r + orr - 1e-9:
                                valid = False
                                break
                        
                        if valid:
                            # Calculate new sum
                            new_sum = best_sum - r + new_r
                            if new_sum > best_sum + 1e-9:
                                best_sum = new_sum
                                best_cells[idx] = [new_x, new_y, new_r]
                                cells = best_cells.copy()
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no improvement, decrease step size
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
    
    # Final validation
    result = best_cells.copy()
    
    # Ensure all circles are within bounds and non-overlapping
    for i in range(N):
        x, y, r = result[i]
        # Clamp to bounds
        r = min(r, x, 1-x, y, 1-y)
        r = max(r, 0.001)
        result[i] = [x, y, r]
    
    # Verify non-overlap
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = result[i]
            xj, yj, rj = result[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-6:
                # Reduce radii to fix overlap
                overlap = ri + rj - dist
                if overlap > 0:
                    ri = max(0.001, ri - overlap/2)
                    rj = max(0.001, rj - overlap/2)
                    result[i] = [xi, yi, ri]
                    result[j] = [xj, yj, rj]
    
    return result
# EVOLVE-BLOCK-END
