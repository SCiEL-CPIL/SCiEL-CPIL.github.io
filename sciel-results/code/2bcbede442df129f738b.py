"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4.
    Maximize the sum of radii.
    
    Strategy: Use a grid-based initialization and then optimize via gradient-free
    optimization (Nelder-Mead) to maximize sum of radii subject to:
    - Non-overlap: distance between centers >= r_i + r_j
    - Bounding box: 0 <= x_i - r_i, x_i + r_i <= W; 0 <= y_i - r_i, y_i + r_i <= H
    - Perimeter: 2*(W + H) <= 4 => W + H <= 2
    
    We parameterize the rectangle aspect ratio and optimize positions/radii.
    """
    # Start with a reasonable grid arrangement
    # For 21 circles, try 5x5 grid minus 4 corners, or 6x4 minus 3
    # Let's use a 5x5 grid, drop 4 circles to get 21
    side = 5
    # Create 5x5 grid positions
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append((i, j))
    
    # We need 21 out of 25. Drop 4 corners for symmetry
    drop = [(0,0), (0,4), (4,0), (4,4)]
    kept = [p for p in positions if p not in drop]
    
    # Initial rectangle: W=1.0, H=1.0 (perimeter 4)
    W = 1.0
    H = 1.0
    
    # Initial positions centered in cells
    centers = []
    for (i, j) in kept:
        cx = (i + 0.5) / side * W
        cy = (j + 0.5) / side * H
        centers.append([cx, cy])
    
    centers = np.array(centers)
    
    # Initial radii: slightly less than half the cell size
    r_init = 0.5 / side * min(W, H) * 0.95
    radii = np.full(N, r_init)
    
    def objective(params):
        """
        params: [W, H, x_0, y_0, r_0, x_1, y_1, r_1, ...]
        Returns negative sum of radii (to minimize) + penalty for violations
        """
        W_val = params[0]
        H_val = params[1]
        
        # Check perimeter constraint
        if 2 * (W_val + H_val) > 4.0 + 1e-10:
            return 1e10
        
        if W_val <= 0 or H_val <= 0:
            return 1e10
        
        penalty = 0.0
        total_r = 0.0
        
        for i in range(N):
            xi = params[2 + 3*i]
            yi = params[3 + 3*i]
            ri = params[4 + 3*i]
            
            if ri <= 0:
                penalty += 1e6 * (1.0 - ri)
                continue
            
            total_r += ri
            
            # Check bounding box
            if xi - ri < -1e-10:
                penalty += 1e6 * (xi - ri + 1e-10)
            if xi + ri > W_val + 1e-10:
                penalty += 1e6 * (xi + ri - W_val - 1e-10)
            if yi - ri < -1e-10:
                penalty += 1e6 * (yi - ri + 1e-10)
            if yi + ri > H_val + 1e-10:
                penalty += 1e6 * (yi + ri - H_val - 1e-10)
        
        # Check non-overlap
        for i in range(N):
            xi = params[2 + 3*i]
            yi = params[3 + 3*i]
            ri = params[4 + 3*i]
            for j in range(i+1, N):
                xj = params[2 + 3*j]
                yj = params[3 + 3*j]
                rj = params[4 + 3*j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist - 1e-10:
                    penalty += 1e6 * (min_dist - dist - 1e-10)
        
        return penalty - total_r
    
    # Build initial parameter vector
    init_params = np.zeros(2 + 3*N)
    init_params[0] = W
    init_params[1] = H
    for i in range(N):
        init_params[2 + 3*i] = centers[i, 0]
        init_params[3 + 3*i] = centers[i, 1]
        init_params[4 + 3*i] = radii[i]
    
    # Bounds
    bounds = []
    # W and H: positive, and W+H <= 2
    bounds.append((1e-6, 2.0))  # W
    bounds.append((1e-6, 2.0))  # H
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((1e-6, 1.0))  # r
    
    # Run Nelder-Mead optimization
    result = minimize(
        objective,
        init_params,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    # Extract solution
    W_opt = result.x[0]
    H_opt = result.x[1]
    
    circles = []
    for i in range(N):
        xi = result.x[2 + 3*i]
        yi = result.x[3 + 3*i]
        ri = max(result.x[4 + 3*i], 1e-6)
        circles.append([xi, yi, ri])
    
    circles = np.array(circles)
    
    # Verify and fix any small violations
    for i in range(N):
        xi, yi, ri = circles[i]
        # Clip to bounding box
        xi = max(ri, min(W_opt - ri, xi))
        yi = max(ri, min(H_opt - ri, yi))
        circles[i] = [xi, yi, ri]
    
    # Final check: if perimeter is violated, scale down
    W_final = circles[:, 0].max() - circles[:, 0].min() + 2 * circles[:, 2].max()
    H_final = circles[:, 1].max() - circles[:, 1].min() + 2 * circles[:, 2].max()
    
    # Actually, let's recompute the bounding box properly
    x_min = (circles[:, 0] - circles[:, 2]).min()
    x_max = (circles[:, 0] + circles[:, 2]).max()
    y_min = (circles[:, 1] - circles[:, 2]).min()
    y_max = (circles[:, 1] + circles[:, 2]).max()
    
    W_box = x_max - x_min
    H_box = y_max - y_min
    
    if 2 * (W_box + H_box) > 4.0:
        scale = 2.0 / (W_box + H_box) * 0.999
        # Scale all radii and re-center
        for i in range(N):
            circles[i, 2] *= scale
        # Re-center
        x_min = (circles[:, 0] - circles[:, 2]).min()
        x_max = (circles[:, 0] + circles[:, 2]).max()
        y_min = (circles[:, 1] - circles[:, 2]).min()
        y_max = (circles[:, 1] + circles[:, 2]).max()
        W_box = x_max - x_min
        H_box = y_max - y_min
        if 2 * (W_box + H_box) > 4.0:
            scale = 2.0 / (W_box + H_box) * 0.999
            for i in range(N):
                circles[i, 2] *= scale
    
    # Shift so that bounding box starts at (0,0)
    x_min = (circles[:, 0] - circles[:, 2]).min()
    y_min = (circles[:, 1] - circles[:, 2]).min()
    circles[:, 0] -= x_min
    circles[:, 1] -= y_min
    
    return circles
# EVOLVE-BLOCK-END
