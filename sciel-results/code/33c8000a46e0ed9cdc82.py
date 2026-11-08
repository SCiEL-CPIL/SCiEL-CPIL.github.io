"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Hexagonal-like initial placement for 26 circles
    # Use a staggered grid pattern
    # Try: 5 rows with 5,6,5,6,4 circles (total 26)
    
    # Define row configurations: (num_circles, row_index)
    # Rows alternate offset
    row_configs = [5, 6, 5, 6, 4]  # 26 total
    
    circles = []
    n_rows = len(row_configs)
    
    # Vertical spacing
    # We'll optimize, but start with reasonable estimates
    # For hexagonal packing, vertical spacing = sqrt(3)/2 * horizontal_spacing
    # Let's estimate: if we have 6 columns max, horizontal spacing ~ 1/6
    # vertical spacing ~ 0.866/6 ~ 0.144
    # 5 rows need ~ 4*0.144 + 2*r = 0.576 + 2r <= 1, so r <= 0.212
    
    # Start with estimated positions
    for i, n_in_row in enumerate(row_configs):
        if i % 2 == 0:
            # Even rows: centered
            x_positions = [(j + 0.5) / n_in_row for j in range(n_in_row)]
        else:
            # Odd rows: offset
            x_positions = [(j + 0.5) / n_in_row for j in range(n_in_row)]
        
        y_pos = (i + 0.5) / n_rows
        
        for x in x_positions:
            circles.append([x, y_pos, 0.15])  # initial radius
    
    circles = np.array(circles[:N])
    
    # Local optimization: iteratively try to increase each circle's radius
    # and adjust positions
    
    def is_valid(circles_arr, tol=1e-6):
        """Check if all circles are non-overlapping and within [0,1]^2."""
        n = len(circles_arr)
        for i in range(n):
            x, y, r = circles_arr[i]
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles_arr[i]
                xj, yj, rj = circles_arr[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True
    
    # Simple local search: for each circle, try to increase radius
    # and move to resolve conflicts
    max_iter = 500
    for iteration in range(max_iter):
        improved = False
        for idx in range(N):
            x, y, r = circles[idx]
            # Try to increase radius slightly
            new_r = r + 0.005
            if new_r > 0.5:
                new_r = 0.5
            
            # Check if increasing radius causes overlap
            can_increase = True
            for jdx in range(N):
                if jdx == idx:
                    continue
                xj, yj, rj = circles[jdx]
                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                if dist < r + rj + 0.001:
                    can_increase = False
                    break
            
            # Check boundary
            if x - new_r < -1e-6 or x + new_r > 1 + 1e-6 or y - new_r < -1e-6 or y + new_r > 1 + 1e-6:
                can_increase = False
            
            if can_increase:
                circles[idx][2] = new_r
                improved = True
        
        # Try to move circles to free up space
        # For each circle, try small perturbations
        for idx in range(N):
            x, y, r = circles[idx]
            # Try moving in 8 directions
            for dx in [-0.02, 0, 0.02]:
                for dy in [-0.02, 0, 0.02]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    # Check boundaries
                    if new_x - r < -1e-6 or new_x + r > 1 + 1e-6:
                        continue
                    if new_y - r < -1e-6 or new_y + r > 1 + 1e-6:
                        continue
                    # Check overlaps
                    valid = True
                    for jdx in range(N):
                        if jdx == idx:
                            continue
                        xj, yj, rj = circles[jdx]
                        dist = np.sqrt((new_x - xj)**2 + (new_y - yj)**2)
                        if dist < r + rj - 1e-6:
                            valid = False
                            break
                    if valid:
                        circles[idx][0] = new_x
                        circles[idx][1] = new_y
                        improved = True
                        break
        
        if not improved:
            break
    
    # Final verification and clipping
    for i in range(N):
        x, y, r = circles[i]
        # Clip to stay in bounds
        r = min(r, x, 1 - x, y, 1 - y)
        circles[i] = [x, y, max(r, 0.01)]
    
    # Ensure validity
    if not is_valid(circles):
        # Fallback: use a safer grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return circles
# EVOLVE-BLOCK-END
