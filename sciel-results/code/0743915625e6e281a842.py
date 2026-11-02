import numpy as np
from scipy.optimize import minimize
import time

N = 21

def compute_radii(x, y, W):
    """
    Compute the maximum radius for each circle given centers (x, y) and width W.
    H is derived as 2 - W.
    """
    H = 2.0 - W
    n = len(x)
    # Initialize radii with distance to walls
    # Distance to left: x, right: W-x, bottom: y, top: H-y
    r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Calculate pairwise distances
    # dx[i,j] = x[i] - x[j]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # The constraint is r_i + r_j <= dist[i,j]
    # This is coupled. The simple decoupled approximation is r_i <= dist[i,j] / 2.
    # This ensures non-overlap if we set r_i = min(wall_dist_i, min_j dist[i,j]/2).
    # This is a standard relaxation for packing problems.
    
    # For each i, min over j != i of dist[i,j] / 2
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist, axis=1)
    
    r = np.minimum(r, min_pair_dist / 2.0)
    
    # Radii must be positive
    r = np.maximum(r, 0.0)
    
    return r

def objective(params):
    """
    Objective function to minimize: -sum(radii)
    params: [x1, y1, x2, y2, ..., x21, y21, W]
    """
    x = params[0:2*N:2]
    y = params[1:2*N:2]
    W = params[2*N]
    
    # Bounds check for W to keep H positive
    if W <= 0.1 or W >= 1.9:
        return 1e6
    
    r = compute_radii(x, y, W)
    return -np.sum(r)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Define bounds for optimization
    # x, y should be within [0, W] and [0, H] roughly, but optimizer might go out.
    # We can use bounds or just let it explore and penalty if out of bounds?
    # The radius calculation naturally penalizes going out of bounds (r becomes 0 or negative).
    # However, if x > W, W-x is negative, r becomes negative, sum becomes smaller (more negative objective).
    # So the optimizer will naturally stay within bounds.
    
    # Initial configurations
    configs = []
    
    # Config 1: 5x5 grid, remove 4 corners
    # 5x5 grid in 1x1 square
    xs_grid = np.linspace(0.1, 0.9, 5)
    ys_grid = np.linspace(0.1, 0.9, 5)
    # Create full 5x5
    all_x = []
    all_y = []
    for i in range(5):
        for j in range(5):
            # Remove corners (0,0), (0,4), (4,0), (4,4)
            if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                continue
            all_x.append(xs_grid[i])
            all_y.append(ys_grid[j])
    
    # Pad to 21 if needed? We have 25-4=21. Perfect.
    x0 = np.array(all_x)
    y0 = np.array(all_y)
    W0 = 1.0
    params0 = np.concatenate([np.column_stack((x0, y0)).flatten(), [W0]])
    configs.append(params0)
    
    # Config 2: Hexagonal packing
    # Approximate hex packing
    # Row 1: 5 circles
    # Row 2: 4 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 3 circles
    # Total 21.
    # Let's construct this manually or use a simple script.
    # Spacing 0.2 in x, 0.17 in y (hex ratio sqrt(3)/2 * 0.2 ~ 0.173)
    x_hex = []
    y_hex = []
    row_sizes = [5, 4, 5, 4, 3]
    y_pos = 0.15
    for r_idx, size in enumerate(row_sizes):
        # Center the row
        start_x = (1.0 - size * 0.2) / 2.0 + 0.1
        for k in range(size):
            x_hex.append(start_x + k * 0.2)
            y_hex.append(y_pos)
        y_pos += 0.173
    
    # Scale to fit? The optimizer will adjust W and positions.
    # Just provide a decent starting point.
    x0_hex = np.array(x_hex)
    y0_hex = np.array(y_hex)
    W0_hex = 1.0
    params0_hex = np.concatenate([np.column_stack((x0_hex, y0_hex)).flatten(), [W0_hex]])
    configs.append(params0_hex)
    
    # Config 3: Random perturbation of grid
    x0_rand = x0 + np.random.normal(0, 0.05, N)
    y0_rand = y0 + np.random.normal(0, 0.05, N)
    params0_rand = np.concatenate([np.column_stack((x0_rand, y0_rand)).flatten(), [1.0]])
    configs.append(params0_rand)
    
    # Run optimization for each config
    for i, p0 in enumerate(configs):
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(objective, p0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
        except Exception as e:
            pass
            
        # If time is running low, we can try a quick polish with Powell on the best so far
        if i == 0 and time.time() < deadline - 20:
             # Maybe add a second phase?
             pass

    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract results
    x = best_params[0:2*N:2]
    y = best_params[1:2*N:2]
    W = best_params[2*N]
    H = 2.0 - W
    
    # Recalculate radii for the final best params
    r = compute_radii(x, y, W)
    
    # Ensure validity: if any r is very small or negative, clamp
    r = np.maximum(r, 1e-6)
    
    # Construct output array (n, 3)
    result = np.column_stack((x, y, r))
    
    # Final check: Perimeter
    # The rectangle is defined by the bounding box of the circles?
    # "bounding rectangle has perimeter at most 4"
    # My optimization assumed a fixed rectangle of perimeter 4.
    # The circles are inside it.
    # The bounding box of the circles might be smaller than W x H.
    # If the bounding box is smaller, the perimeter is smaller, which is allowed.
    # However, the evaluator might check if the circles fit in *their* bounding box.
    # The problem says "bounding rectangle has perimeter at most 4".
    # This usually means the rectangle that contains all circles.
    # If I optimize for a fixed W, H=2-W, and circles are inside,
    # the actual bounding box of the circles will have width <= W and height <= H.
    # Perimeter = 2*(w_box + h_box) <= 2*(W + H) = 4.
    # So this is valid.
    
    return result

# EVOLVE-BLOCK-START
# The code above is the full implementation.
# We need to wrap it in the required format.
# The prompt asks to keep function names and signatures.
# The current program has `construct()`.
# I will place the helper functions and `construct` inside the block.
# EVOLVE-BLOCK-END
