import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    # Initialization: Jittered grid
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of side x side
    xs = np.linspace(0.5/side, 1 - 0.5/side, side)
    ys = np.linspace(0.5/side, 1 - 0.5/side, side)
    
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Select first N points
    x = grid_x[:N].copy()
    y = grid_y[:N].copy()
    
    # Initial radii
    r = np.full(N, 0.5 / side * 0.8)
    
    # Jitter
    np.random.seed(42)
    x += np.random.uniform(-0.01, 0.01, N)
    y += np.random.uniform(-0.01, 0.01, N)
    x = np.clip(x, 0.01, 0.99)
    y = np.clip(y, 0.01, 0.99)
    
    # Optimization parameters
    steps = 5000
    lr = 0.02
    overlap_penalty = 500.0
    boundary_penalty = 500.0
    
    for step in range(steps):
        # Calculate distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlap = r_sum - d
        overlap_pos = np.maximum(0, overlap)
        
        # Overlap Gradients
        W = np.zeros((N, N))
        mask = overlap > 0
        W[mask] = 2 * overlap_pos[mask] / d[mask]
        
        grad_x_overlap = np.sum(W * dx, axis=1)
        grad_y_overlap = np.sum(W * dy, axis=1)
        grad_r_overlap = np.sum(2 * overlap_pos, axis=1)
        
        # Boundary Penalties
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, r - (1 - x))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, r - (1 - y))
        
        grad_x_bound = -2 * viol_left + 2 * viol_right
        grad_y_bound = -2 * viol_bottom + 2 * viol_top
        grad_r_bound = 2 * (viol_left + viol_right + viol_bottom + viol_top)
        
        # Total Gradients
        grad_x = overlap_penalty * grad_x_overlap + boundary_penalty * grad_x_bound
        grad_y = overlap_penalty * grad_y_overlap + boundary_penalty * grad_y_bound
        grad_r = -1.0 + overlap_penalty * grad_r_overlap + boundary_penalty * grad_r_bound
        
        # Update
        x -= lr * grad_x
        y -= lr * grad_y
        r -= lr * grad_r
        
        r = np.maximum(r, 1e-6)
        
    # Safety Net: Shrink to ensure validity
    for _ in range(20):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        
        max_overlap = np.max(r_sum - d)
        max_viol_left = np.max(r - x)
        max_viol_right = np.max(r - (1 - x))
        max_viol_bottom = np.max(r - y)
        max_viol_top = np.max(r - (1 - y))
        
        # Required shrinkage
        # For overlap: need r_i + r_j - d <= 0 => 2*shrink >= max_overlap => shrink >= max_overlap/2
        # For boundary: need r_i - x_i <= 0 => shrink >= max_viol
        shrink = max(max_overlap / 2.0, max_viol_left, max_viol_right, max_viol_bottom, max_viol_top)
        
        if shrink <= 1e-7:
            break
            
        r -= shrink + 1e-8
        r = np.maximum(r, 1e-8)
        
    return np.column_stack((x, y, r))
# EVOLVE-BLOCK-END
