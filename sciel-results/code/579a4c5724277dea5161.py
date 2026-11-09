import numpy as np
import time

N = 26

def compute_energy(x, y, r, alpha, beta):
    """
    x, y: (N,)
    r: (N,)
    alpha: weight for overlap penalty
    beta: weight for boundary penalty
    """
    # Sum of radii (we want to maximize, so minimize -sum)
    obj = -np.sum(r)
    
    # Overlap penalty
    # dist matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_dist - dist)
    # Zero out diagonal
    np.fill_diagonal(overlap, 0)
    overlap_penalty = np.sum(overlap**2)
    
    # Boundary penalty
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    bx1 = np.maximum(0, r - x)
    bx2 = np.maximum(0, r - (1 - x))
    by1 = np.maximum(0, r - y)
    by2 = np.maximum(0, r - (1 - y))
    bound_penalty = np.sum(bx1**2) + np.sum(bx2**2) + np.sum(by1**2) + np.sum(by2**2)
    
    return obj + alpha * overlap_penalty + beta * bound_penalty

def compute_grad(x, y, r, alpha, beta):
    """
    Returns gradients for x, y, r
    """
    N = len(r)
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Gradient of -sum(r)
    gr -= 1.0
    
    # Overlap gradient
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    # overlap = max(0, min_dist - dist)
    # grad_overlap w.r.t x_i: sum_j 2 * overlap_ij * (-(x_i - x_j) / dist_ij)
    # Note: overlap is 0 if dist > min_dist.
    
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    # Unit vector from j to i
    # ux = dx / dist
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    # d(overlap)/d(x_i) = sum_j 2 * overlap_ij * (-ux_ij)
    # But only where overlap > 0
    # Let's compute the force vector
    # Force on i from j is proportional to overlap * (x_i - x_j) / dist
    # This pushes i away from j.
    
    # grad_x_i = -2 * alpha * sum_j overlap_ij * ux_ij
    # Wait, energy is alpha * sum overlap^2.
    # d/dx_i (overlap_ij^2) = 2 * overlap_ij * d(overlap_ij)/dx_i
    # overlap_ij = r_i + r_j - dist_ij
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij = ux_ij
    # d(overlap_ij)/dx_i = -ux_ij
    # So term is 2 * overlap_ij * (-ux_ij)
    
    gx -= 2.0 * alpha * np.sum(overlap * ux, axis=1)
    gy -= 2.0 * alpha * np.sum(overlap * uy, axis=1)
    
    # Gradient w.r.t r_i
    # d(overlap_ij)/dr_i = 1
    # So term is 2 * overlap_ij * 1
    # Sum over j
    gr += 2.0 * alpha * np.sum(overlap, axis=1)
    
    # Boundary gradient
    # bx1 = max(0, r - x)
    # if r > x: grad_x = -2*beta*(r-x), grad_r = 2*beta*(r-x)
    # bx2 = max(0, r - (1-x))
    # if r > 1-x: grad_x = 2*beta*(r-(1-x)), grad_r = 2*beta*(r-(1-x))
    
    bx1 = np.maximum(0, r - x)
    bx2 = np.maximum(0, r - (1 - x))
    by1 = np.maximum(0, r - y)
    by2 = np.maximum(0, r - (1 - y))
    
    gx += 2.0 * beta * (bx2 - bx1)
    gy += 2.0 * beta * (by2 - by1)
    gr += 2.0 * beta * (bx1 + bx2 + by1 + by2)
    
    return gx, gy, gr

def is_valid(x, y, r, tol=1e-5):
    if np.any(r < 0): return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol): return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol): return False
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_config = None
    
    # Initialize with a grid
    side = int(np.ceil(np.sqrt(N)))
    # Use a 6x6 grid for 26 circles? 
    # Let's just use a 5x5 grid (25 circles) and place the 26th in the center of the largest gap?
    # Or simpler: 6x6 grid, take first 26.
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N: break
            xs.append((i + 0.5) / side)
            ys.append((j + 0.5) / side)
            rs.append(0.5 / side * 0.9) # Slightly smaller to start
            count += 1
        
    x = np.array(xs)
    y = np.array(ys)
    r = np.array(rs)
    
    # Optimization parameters
    alpha = 10.0
    beta = 10.0
    lr = 0.01
    
    # Multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        # Perturb initial state
        if restart > 0:
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.02, 0.05, N)
        
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Run gradient descent
        for step in range(2000):
            if time.time() > deadline - 5:
                break
                
            gx, gy, gr = compute_grad(x, y, r, alpha, beta)
            
            # Adaptive learning rate? Simple decay
            current_lr = lr / (1 + step * 0.001)
            
            x += -current_lr * gx
            y += -current_lr * gy
            r += -current_lr * gr
            
            # Project r to positive
            r = np.maximum(r, 0.001)
            
            # Project x, y to [0,1] roughly to keep them in bounds
            # The penalty handles it, but clipping helps stability
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Check validity and update best
            if is_valid(x, y, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_config = (x.copy(), y.copy(), r.copy())
                    
            # Increase penalty weights to enforce constraints more strictly
            if step % 100 == 0:
                alpha *= 1.5
                beta *= 1.5
                # Reset learning rate slightly?
                # lr = 0.01
                
    # Final check
    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
