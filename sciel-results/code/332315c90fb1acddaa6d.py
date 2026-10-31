import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial configuration: Grid
    # 21 circles. 4x5 grid or 5x4.
    # Let's use a 4x5 grid (4 rows, 5 cols) or 5x4.
    # Aspect ratio 1:1 might be good start.
    W_init = 1.0
    H_init = 1.0
    
    rows, cols = 4, 5
    # Adjust to fit 21: 4*5=20, add 1 in center? Or just 5x5 grid and remove 4?
    # Let's use a 5x5 grid and take the first 21.
    side = 5
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Place in grid
    for i in range(N):
        row = i // side
        col = i % side
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.5 / side - 0.01
    
    W = W_init
    H = H_init
    
    # Optimization variables
    # We will optimize x, y, r, W
    # Loss function
    def loss_and_grad(x, y, r, W):
        H = 2.0 - W
        # Overlap penalty
        # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # We need vectorized distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        
        # Only upper triangle to avoid double counting
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap = overlap[mask]
        
        loss_overlap = np.sum(overlap**2)
        
        # Gradient of overlap penalty
        # d/dx_i sum_{j>i} max(0, r_i+r_j-d_ij)^2
        # Let u_ij = r_i+r_j-d_ij
        # d/dx_i u_ij^2 = 2 u_ij * d/dx_i (r_i+r_j-d_ij) = 2 u_ij * (-d/dx_i d_ij) = 2 u_ij * (x_i-x_j)/d_ij
        # Note: if d_ij is 0, gradient is undefined, but overlap is large, force should be strong.
        # We can add a small epsilon to d in denominator.
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        grad_W = 0.0
        
        # Iterate over pairs for gradient (N=21, 210 pairs, fast enough)
        for i in range(N):
            for j in range(i+1, N):
                if overlap_val := (r[i] + r[j] - d[i,j]):
                    if overlap_val > 0:
                        d_ij = d[i,j]
                        if d_ij < 1e-6:
                            # Random direction
                            angle = np.random.uniform(0, 2*np.pi)
                            dx_dir = np.cos(angle)
                            dy_dir = np.sin(angle)
                        else:
                            dx_dir = (x[i] - x[j]) / d_ij
                            dy_dir = (y[i] - y[j]) / d_ij
                        
                        # Gradient of 0.5 * overlap^2 (factor 1/2 cancels with 2 in derivative)
                        # d/dx_i (0.5 * u^2) = u * du/dx_i = u * (x_i-x_j)/d_ij
                        force_mag = overlap_val
                        
                        grad_x[i] += force_mag * dx_dir
                        grad_y[i] += force_mag * dy_dir
                        grad_x[j] -= force_mag * dx_dir
                        grad_y[j] -= force_mag * dy_dir
                        grad_r[i] += force_mag
                        grad_r[j] += force_mag
        
        # Boundary penalty
        # x_i < r_i  -> max(0, r_i - x_i)^2
        # x_i > W - r_i -> max(0, x_i - (W-r_i))^2
        # y_i < r_i
        # y_i > H - r_i
        
        # Left
        viol_left = np.maximum(0, r - x)
        loss_bound = np.sum(viol_left**2)
        grad_x -= 2 * viol_left
        
        # Right
        viol_right = np.maximum(0, x - (W - r))
        loss_bound += np.sum(viol_right**2)
        grad_x += 2 * viol_right
        # d/dW of viol_right^2: d/dW max(0, x - W + r)^2 = 2 max(0, x-W+r) * (-1)
        grad_W -= 2 * np.sum(viol_right)
        
        # Bottom
        viol_bottom = np.maximum(0, r - y)
        loss_bound += np.sum(viol_bottom**2)
        grad_y -= 2 * viol_bottom
        
        # Top
        viol_top = np.maximum(0, y - (H - r))
        loss_bound += np.sum(viol_top**2)
        grad_y += 2 * viol_top
        # d/dW of viol_top^2: H = 2-W. d/dW max(0, y - (2-W) + r)^2 = 2 max(0, y - 2 + W + r) * (1)
        grad_W += 2 * np.sum(viol_top)
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        loss_obj = -np.sum(r)
        grad_r -= 1.0
        
        total_loss = loss_overlap + loss_bound + loss_obj
        total_grad_x = grad_x
        total_grad_y = grad_y
        total_grad_r = grad_r
        total_grad_W = grad_W
        
        return total_loss, total_grad_x, total_grad_y, total_grad_r, total_grad_W

    # Gradient Descent
    lr = 0.01
    best_sum_r = 0
    best_state = None
    
    # Run optimization
    for step in range(100000):
        if time.time() - start_time > time_limit:
            break
            
        loss, gx, gy, gr, gW = loss_and_grad(x, y, r, W)
        
        # Update
        x -= lr * gx
        y -= lr * gy
        r -= lr * gr
        W -= lr * gW
        
        # Constrain W
        W = np.clip(W, 0.5, 1.5)
        
        # Constrain r to be positive
        r = np.maximum(r, 0.01)
        
        # Check current sum
        current_sum = np.sum(r)
        if current_sum > best_sum_r:
            best_sum_r = current_sum
            best_state = (x.copy(), y.copy(), r.copy(), W)
            
        # Adaptive learning rate?
        # If loss increases, reduce lr?
        
    if best_state is None:
        best_state = (x, y, r, W)
        
    bx, by, br, bW = best_state
    bH = 2.0 - bW
    
    # Post-processing: Ensure validity
    # Find max scale s such that circles are disjoint and in bounds
    # We want max s such that:
    # 1. s*br[i] <= bx[i] and bx[i] + s*br[i] <= bW
    # 2. s*br[i] <= by[i] and by[i] + s*br[i] <= bH
    # 3. dist(i,j) >= s*(br[i] + br[j])
    
    # Bound constraints on s:
    # s <= bx[i] / br[i]
    # s <= (bW - bx[i]) / br[i]
    # s <= by[i] / br[i]
    # s <= (bH - by[i]) / br[i]
    
    s_bound = np.inf
    for i in range(N):
        if br[i] > 0:
            s_bound = min(s_bound, bx[i] / br[i], (bW - bx[i]) / br[i])
            s_bound = min(s_bound, by[i] / br[i], (bH - by[i]) / br[i])
            
    if s_bound < 0.1:
        # Fallback to small radii if bounds are tight
        s_bound = 0.1
        
    # Overlap constraints on s:
    # s <= d_ij / (br[i] + br[j])
    s_overlap = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d_ij = np.sqrt((bx[i]-bx[j])**2 + (by[i]-by[j])**2)
            sum_r = br[i] + br[j]
            if sum_r > 0:
                s_overlap = min(s_overlap, d_ij / sum_r)
                
    s = min(s_bound, s_overlap, 1.0)
    
    # If s is very small, the packing is bad.
    # We can try to recover by shrinking more.
    # The value is sum(s * br)
    
    final_r = s * br
    final_x = bx
    final_y = by
    
    # Construct output
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = final_x[i]
        result[i, 1] = final_y[i]
        result[i, 2] = final_r[i]
        
    return result
