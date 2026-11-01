"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Grid
    # Generate a dense hexagonal grid and select the 32 points closest to the center
    hex_points = []
    # Use a range large enough to cover the unit square
    for q in range(-10, 11):
        for r in range(-10, 11):
            # Axial to Cartesian (pointy top)
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    
    # Center the points around (0.5, 0.5)
    center = np.array([0.5, 0.5])
    hex_points += (center - np.mean(hex_points, axis=0))
    
    # Select 32 points closest to center
    dists = np.sum((hex_points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = hex_points[sorted_idx[:N]]
    
    # Initial radii: small value to start
    init_radii = np.full(N, 0.02)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    x0[0::3] = init_points[:, 0]
    x0[1::3] = init_points[:, 1]
    x0[2::3] = init_radii
    
    # Penalty weight for constraints
    C = 100.0
    
    def compute_gradient_and_penalty(vars):
        """
        Compute gradient of the penalty function and the penalty value.
        Objective to minimize: -sum(r) + C * sum(violations^2)
        """
        grad = np.zeros(3 * N)
        penalty_val = 0.0
        
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # 1. Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # 2. Pairwise Overlap Penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq)
                r_sum = r[i] + r[j]
                
                if d < 1e-9:
                    dx = 1e-9
                    d = 1e-9
                    d_sq = 1e-18
                
                if d < r_sum:
                    V = r_sum - d
                    penalty_val += V * V
                    coeff = 2.0 * C * V
                    inv_d = 1.0 / d
                    
                    # Grads for i
                    grad[3*i + 0] += coeff * (-dx * inv_d)
                    grad[3*i + 1] += coeff * (-dy * inv_d)
                    grad[3*i + 2] += coeff * 1.0
                    
                    # Grads for j
                    grad[3*j + 0] += coeff * (dx * inv_d)
                    grad[3*j + 1] += coeff * (dy * inv_d)
                    grad[3*j + 2] += coeff * 1.0

        # 3. Boundary Penalties
        for i in range(N):
            # Left: x_i - r_i >= 0
            if x[i] < r[i]:
                V = r[i] - x[i]
                penalty_val += V * V
                coeff = 2.0 * C * V
                grad[3*i + 0] += coeff * (-1.0)
                grad[3*i + 2] += coeff * 1.0
                
            # Right: 1 - x_i - r_i >= 0
            if x[i] + r[i] > 1.0:
                V = x[i] + r[i] - 1.0
                penalty_val += V * V
                coeff = 2.0 * C * V
                grad[3*i + 0] += coeff * (1.0)
                grad[3*i + 2] += coeff * 1.0
                
            # Bottom: y_i - r_i >= 0
            if y[i] < r[i]:
                V = r[i] - y[i]
                penalty_val += V * V
                coeff = 2.0 * C * V
                grad[3*i + 1] += coeff * (-1.0)
                grad[3*i + 2] += coeff * 1.0
                
            # Top: 1 - y_i - r_i >= 0
            if y[i] + r[i] > 1.0:
                V = y[i] + r[i] - 1.0
                penalty_val += V * V
                coeff = 2.0 * C * V
                grad[3*i + 1] += coeff * (1.0)
                grad[3*i + 2] += coeff * 1.0
                
        return grad, penalty_val

    def is_valid(vars, tol=1e-6):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        for i in range(N):
            if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
                return False
            if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
                return False
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    # Gradient Descent Loop
    best_val = -np.sum(init_radii)
    best_vars = x0.copy()
    
    lr = 0.05
    # We will run until time is up or penalty is very low
    steps = 5000
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        g, pen = compute_gradient_and_penalty(x0)
        
        # Normalize gradient to prevent large steps if penalty is high
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-9:
            x0 -= lr * (g / g_norm) * min(g_norm, 1.0) # Scaled step
        else:
            break
            
        # Simple Projection: 
        # Ensure radii are positive
        x0[2::3] = np.maximum(x0[2::3], 0.01)
        # Clip positions to [0, 1] to keep them in the box roughly
        x0[0::3] = np.clip(x0[0::3], 0.0, 1.0)
        x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
        
        # Evaluate current sum of radii
        current_sum = np.sum(x0[2::3])
        if current_sum > best_val:
            best_val = current_sum
            best_vars = x0.copy()
            
        # Decay learning rate
        lr *= 0.99
        
        # If penalty is very low, we can try to increase radii slightly more aggressively
        # by reducing the penalty weight influence, but the gradient method handles this.
        
    # Final Check: Validate best_vars
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    if not is_valid(best_vars):
        # Fallback to seed if optimization failed to converge to valid state
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Return array (x, y, radius)
    result = np.column_stack((x_best, y_best, r_best))
    return result
# EVOLVE-BLOCK-END
