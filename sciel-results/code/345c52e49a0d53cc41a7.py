import numpy as np
import time
import math

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Define a few initial configurations
    # 1. 3x7 Grid
    # 2. 4x6 Grid (21 circles, 3 removed)
    # 3. 5x5 Grid (21 circles, 4 removed)
    
    configs = []
    
    # Config 1: 3 rows, 7 cols
    r0 = 0.1
    w0 = 7 * 2 * r0
    h0 = 3 * 2 * r0
    # Adjust w, h to perimeter 4
    # w+h=2. 
    # If we start with a grid, we can scale it.
    # But we will optimize w as well.
    
    # Helper to create a grid config
    def make_grid(rows, cols, n_circles):
        # Create grid of all positions
        # Remove 'excess' circles from the center or corners to get n_circles
        # For simplicity, just take the first n_circles in row-major order
        # This might create a block shape
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) < n_circles:
                    pts.append((c, r))
                else:
                    break
            if len(pts) >= n_circles:
                break
        
        # Center the grid
        max_c = max(p[0] for p in pts)
        max_r = max(p[1] for p in pts)
        
        # Initial radius
        # Estimate based on perimeter
        # w ~ (max_c+1)*2r, h ~ (max_r+1)*2r
        # 2r(max_c+1 + max_r+1) = 2 => r = 1 / (max_c+1 + max_r+1)
        r_est = 1.0 / (max_c + 1 + max_r + 1)
        
        x = []
        y = []
        rad = []
        for p in pts:
            x.append((p[0] + 0.5) * 2 * r_est)
            y.append((p[1] + 0.5) * 2 * r_est)
            rad.append(r_est)
            
        w_init = (max_c + 1) * 2 * r_est
        h_init = (max_r + 1) * 2 * r_est
        
        return np.array(x), np.array(y), np.array(rad), w_init, h_init

    # Generate configs
    # 3x7 is exactly 21
    x1, y1, r1, w1, h1 = make_grid(3, 7, 21)
    configs.append((x1, y1, r1, w1, h1))
    
    # 4x6 (24 slots, take 21)
    x2, y2, r2, w2, h2 = make_grid(4, 6, 21)
    configs.append((x2, y2, r2, w2, h2))
    
    # 5x5 (25 slots, take 21)
    x3, y3, r3, w3, h3 = make_grid(5, 5, 21)
    configs.append((x3, y3, r3, w3, h3))
    
    # 2x11 (22 slots, take 21)
    x4, y4, r4, w4, h4 = make_grid(2, 11, 21)
    configs.append((x4, y4, r4, w4, h4))
    
    # Optimization function
    def optimize(x, y, r, w, h, max_steps=2000):
        # State: x, y, r, w
        # h is derived: 2 - w
        
        # Flatten to vector for gradient computation if needed, 
        # but we can do manual gradient steps with NumPy arrays.
        
        x = x.copy()
        y = y.copy()
        r = r.copy()
        w = w
        
        # Penalty weights
        alpha = 100.0
        beta = 100.0
        
        # Learning rate
        lr = 0.01
        
        best_local_score = -np.inf
        best_local_state = None
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            h = 2.0 - w
            
            # 1. Calculate Overlaps
            # d_ij = dist(i, j)
            # overlap_ij = max(0, r_i + r_j - d_ij)
            
            # Vectorized distance matrix
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Fill diagonal with inf to avoid self-overlap
            np.fill_diagonal(dist, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - dist)
            
            # Gradient of overlap w.r.t x_i
            # d(overlap)/dx_i = - (x_i - x_j) / d_ij * (1 if overlap > 0 else 0)
            # We need to sum over j
            
            # Mask for overlapping pairs
            mask = overlap > 0
            
            # Gradients for x
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            grad_r = np.zeros(N)
            
            # Only compute gradients for overlapping pairs to save time?
            # With N=21, N^2=441, which is small. We can do it all.
            
            # dist is N x N
            # To avoid division by zero, use dist where mask is true
            # For non-overlapping, grad is 0.
            
            # We need to handle the case where dist is 0 (should not happen with init)
            safe_dist = np.where(mask, dist, 1.0)
            
            # Unit vectors
            ux = dx / safe_dist
            uy = dy / safe_dist
            
            # Gradient contribution from overlap
            # L_overlap = sum(overlap^2)
            # dL/dx_i = sum_j 2 * overlap_ij * d(overlap_ij)/dx_i
            # d(overlap_ij)/dx_i = - ux_ij (if i is the first index in pair? No, symmetric)
            
            # Let's sum over all j for each i.
            # grad_x[i] = sum_j 2 * overlap[i,j] * (-ux[i,j])
            
            # Using broadcasting
            # overlap is N x N. ux is N x N.
            # grad_x = -2 * np.sum(overlap * ux, axis=1)
            
            grad_x_overlap = -2.0 * np.sum(overlap * ux, axis=1)
            grad_y_overlap = -2.0 * np.sum(overlap * uy, axis=1)
            
            # Gradient w.r.t r_i
            # d(overlap_ij)/dr_i = 1 (if overlap > 0)
            # grad_r[i] = sum_j 2 * overlap[i,j] * 1
            grad_r_overlap = 2.0 * np.sum(overlap, axis=1)
            
            # 2. Boundary Violations
            # x_i < r_i  => viol = r_i - x_i
            # x_i > w - r_i => viol = x_i - (w - r_i)
            # y_i < r_i  => viol = r_i - y_i
            # y_i > h - r_i => viol = y_i - (h - r_i)
            
            viol_x_min = np.maximum(0, r - x)
            viol_x_max = np.maximum(0, x - (w - r))
            viol_y_min = np.maximum(0, r - y)
            viol_y_max = np.maximum(0, y - (h - r))
            
            # L_boundary = sum(viol^2)
            # dL/dx_i = -2*viol_x_min + 2*viol_x_max
            # dL/dy_i = -2*viol_y_min + 2*viol_y_max
            # dL/dr_i = 2*viol_x_min - 2*viol_x_max - 2*viol_y_min + 2*viol_y_max
            # dL/dw = -2*viol_x_max (since w is in x_max term: x - (w-r) = x - w + r. d/dw = -1)
            
            grad_x_bound = -2.0 * viol_x_min + 2.0 * viol_x_max
            grad_y_bound = -2.0 * viol_y_min + 2.0 * viol_y_max
            grad_r_bound = 2.0 * viol_x_min - 2.0 * viol_x_max - 2.0 * viol_y_min + 2.0 * viol_y_max
            grad_w_bound = -2.0 * np.sum(viol_x_max)
            
            # 3. Objective: Maximize sum(r) => Minimize -sum(r)
            # grad_obj_r = -1.0
            # grad_obj_x = 0
            # grad_obj_y = 0
            # grad_obj_w = 0
            
            # Total Gradients
            # L = -sum(r) + alpha * L_overlap + beta * L_boundary
            # Note: L_overlap is sum of squares, so we included the 2 in gradient.
            # Wait, if L = sum(overlap^2), dL = 2*overlap * d(overlap).
            # My grad calculations above used 2.0 * ... which is correct for sum of squares.
            
            grad_x_total = alpha * grad_x_overlap + beta * grad_x_bound
            grad_y_total = alpha * grad_y_overlap + beta * grad_y_bound
            grad_r_total = -1.0 + alpha * grad_r_overlap + beta * grad_r_bound
            grad_w_total = beta * grad_w_bound
            
            # Update
            x -= lr * grad_x_total
            y -= lr * grad_y_total
            r -= lr * grad_r_total
            w -= lr * grad_w_total
            
            # Constraints on variables
            # r > 0
            r = np.maximum(r, 1e-6)
            # w in [0.1, 1.9] to keep h positive and reasonable
            w = np.clip(w, 0.1, 1.9)
            
            # Check validity and score
            h = 2.0 - w
            # Check if all circles are inside
            # x in [r, w-r], y in [r, h-r]
            # dist >= r_i + r_j
            
            valid = True
            # Boundary check
            if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
                valid = False
            # Overlap check
            # We can reuse dist and r_sum from earlier in loop? 
            # No, x,y,r changed. Recalculate.
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            r_sum = r[:, None] + r[None, :]
            if np.any(dist < r_sum - 1e-9):
                valid = False
            
            score = np.sum(r)
            if valid and score > best_local_score:
                best_local_score = score
                best_local_state = (x.copy(), y.copy(), r.copy(), w, h)
                
        return best_local_state, best_local_score
    
    # Run optimization for each config
    for i, (x, y, r, w, h) in enumerate(configs):
        if time.time() > deadline:
            break
        state, score = optimize(x, y, r, w, h)
        if state and score > best_score:
            best_score = score
            best_result = state
            
    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Return seed-like result
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x, y, r, w, h = best_result
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-START
# The code above is the complete program. 
# The construct function is the entry point.
# EVOLVE-BLOCK-END
