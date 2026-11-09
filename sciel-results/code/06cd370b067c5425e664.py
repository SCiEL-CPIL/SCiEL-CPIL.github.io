import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_config = None
    
    # Multiple restarts
    num_restarts = 20
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Random initialization with small radii
        np.random.seed(42 + restart)
        
        # Grid-like initialization often works better than pure random for packing
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) < N:
                    # Jittered grid
                    x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    # Keep within [0.1, 0.9] initially
                    x = np.clip(x, 0.1, 0.9)
                    y = np.clip(y, 0.1, 0.9)
                    xs.append(x)
                    ys.append(y)
                    rs.append(0.05) # Start with small radius
        
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Gradient Descent Parameters
        lr = 0.01
        lambda_overlap = 10.0
        lambda_bound = 10.0
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Calculate Distances
            # diff[i, j] = x_i - x_j
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero in gradients
            d_safe = np.where(d < 1e-8, 1e-8, d)
            
            # Overlap Penalty
            # P_overlap = sum_{i<j} max(0, r_i + r_j - d_ij)^2
            # Gradient w.r.t x_i: -2 * max(0, r_i + r_j - d_ij) * (x_i - x_j) / d_ij
            # We sum over j != i
            
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d)
            
            # Mask diagonal to ignore self-overlap
            np.fill_diagonal(overlap, 0)
            
            # Gradients for Overlap
            # dP/dx_i = -2 * sum_j overlap_ij * (x_i - x_j) / d_ij
            # Note: overlap is symmetric.
            
            grad_x_overlap = -2 * lambda_overlap * np.sum(overlap * (dx / d_safe), axis=1)
            grad_y_overlap = -2 * lambda_overlap * np.sum(overlap * (dy / d_safe), axis=1)
            
            # Gradient w.r.t r_i from overlap:
            # dP/dr_i = 2 * sum_j overlap_ij * 1 (since d(r_i+r_j)/dr_i = 1)
            # Wait, P = (r_i + r_j - d)^2. dP/dr_i = 2(r_i+r_j-d)*1.
            grad_r_overlap = 2 * lambda_overlap * np.sum(overlap, axis=1)
            
            # Boundary Penalty
            # P_bound = sum_i [ max(0, r_i - x_i)^2 + max(0, r_i - (1-x_i))^2 + ... ]
            # Let's compute terms
            # Term 1: max(0, r - x)^2
            # d/dr = 2*max(0, r-x)
            # d/dx = -2*max(0, r-x)
            
            bound_x_left = np.maximum(0, r - x)
            bound_x_right = np.maximum(0, r - (1 - x))
            bound_y_bottom = np.maximum(0, r - y)
            bound_y_top = np.maximum(0, r - (1 - y))
            
            grad_x_bound = -2 * lambda_bound * bound_x_left + 2 * lambda_bound * bound_x_right
            grad_y_bound = -2 * lambda_bound * bound_y_bottom + 2 * lambda_bound * bound_y_top
            
            grad_r_bound = 2 * lambda_bound * (bound_x_left + bound_x_right + bound_y_bottom + bound_y_top)
            
            # Objective: Maximize Sum(r) -> Minimize -Sum(r)
            # d(-Sum(r))/dr = -1
            grad_r_obj = -np.ones(N)
            
            # Total Gradients
            grad_x = grad_x_overlap + grad_x_bound
            grad_y = grad_y_overlap + grad_y_bound
            grad_r = grad_r_overlap + grad_r_bound + grad_r_obj
            
            # Update
            x -= lr * grad_x
            y -= lr * grad_y
            r -= lr * grad_r
            
            # Clip radii to be positive
            r = np.maximum(r, 0.01)
            
            # Optional: Clip positions to [0,1] to prevent flying off
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Decay learning rate
            lr *= 0.999
            
        # Post-processing: Ensure validity
        # 1. Clamp radii to boundary constraints
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        
        # 2. Check overlaps and shrink if necessary
        # We can iteratively shrink radii if overlaps exist
        # This is a simple repair strategy
        valid = False
        for _ in range(100):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            min_dist = np.min(d, axis=1)
            required_dist = r + np.max(r) # This is not correct for pairwise.
            
            # Proper check:
            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                valid = True
                break
            
            # If overlap, shrink all radii by 0.99
            r *= 0.99
            
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
                
    if best_config is None:
        # Fallback to seed if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
