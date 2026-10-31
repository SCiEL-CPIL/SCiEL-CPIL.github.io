"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Maximize sum of radii.
    
    Strategy:
    1. Use a grid-based initialization to get a valid starting configuration.
    2. Use gradient-based optimization to maximize the sum of radii while maintaining
       disjoint interiors and bounding box perimeter <= 4.
    3. The variables are (x_i, y_i, r_i) for each circle i.
    4. Constraints:
       - x_i in [r_i, W - r_i]
       - y_i in [r_i, H - r_i]
       - r_i > 0
       - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
       - 2*(W + H) <= 4, i.e., W + H <= 2
       - W = max(2*max(x_i), 2*max(r_i)) ... actually W >= max(x_i + r_i) - min(x_i - r_i)
         Let's define the bounding box properly.
    
    Actually, let's think about the bounding box. The circles must be contained in a rectangle
    of perimeter 4. The rectangle can be positioned anywhere, but WIC (Width) and HEIGHT (H)
    satisfy 2*(W + H) <= 4 => W + H <= 2.
    
    We can fix the rectangle to be [0, W] x [0, H] with W + H <= 2.
    Then each circle (x_i, y_i, r_i) must satisfy:
      r_i <= x_i <= W - r_i
      r_i <= y_i <= H - r_i
      (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    Objective: maximize sum(r_i)
    
    This is a non-convex problem. We'll use scipy.optimize.minimize with method='SLSQP'
    or 'trust-constr'. Given the non-convexity, we need good initialization.
    
    Let's try a different approach: parameterize by the rectangle dimensions and circle positions.
    
    Alternative: Use a penalty method or barrier method to handle the constraints.
    
    Let's use scipy.optimize.minimize with the following:
    - Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_20, y_20, r_20, W, H]
    - Objective: -sum(r_i)  (minimize negative sum)
    - Constraints:
      1. W + H <= 2
      2. For each i: x_i - r_i >= 0, W - x_i - r_i >= 0, y_i - r_i >= 0, H - y_i - r_i >= 0
      3. For each i != j: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
      4. r_i >= epsilon (small positive)
    
    The number of variables is 21*3 + 2 = 65.
    The number of constraints is 1 + 21*4 + C(21,2) = 1 + 84 + 210 = 295.
    
    This is feasible for SLSQP, but might be slow. Let's try it.
    
    For initialization, we can use a grid packing. Let's use a 5x5 grid (25 cells) and pick 21.
    Set W = H = 1.0 (so perimeter = 4). Place circles at grid points with small radii.
    """
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initialization: 5x5 grid, W=H=1
    side = 5
    W_init = 1.0
    H_init = 1.0
    r_init = 0.5 / side - 0.01  # Slightly smaller to avoid constraint violation
    
    # Create initial positions
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side * W_init
            y = (j + 0.5) / side * H_init
            positions.append([x, y, r_init])
    
    # Take first 21
    positions = positions[:N]
    
    # Variables: [x_0, y_0, r_0, ..., x_20, y_20, r_20, W, H]
    x0 = []
    for pos in positions:
        x0.extend(pos)
    x0.extend([W_init, H_init])
    x0 = np.array(x0)
    
    n_vars = N * 3 + 2
    W_idx = n_vars - 2
    H_idx = n_vars - 1
    
    def objective(x):
        radii = x[2::3][:N]
        return -np.sum(radii)
    
    def gradient(x):
        grad = np.zeros(n_vars)
        grad[2::3][:N] = -1.0
        return grad
    
    # Constraints
    def constraint_perimeter(x):
        """W + H <= 2 => 2 - (W + H) >= 0"""
        W = x[W_idx]
        H = x[H_idx]
        return 2.0 - (W + H)
    
    def constraint_box(x):
        """All circles inside box: x_i - r_i >= 0, W - x_i - r_i >= 0, etc."""
        W = x[W_idx]
        H = x[H_idx]
        cons = []
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            cons.append(xi - ri)
            cons.append(W - xi - ri)
            cons.append(yi - ri)
            cons.append(H - yi - ri)
        return np.array(cons)
    
    def constraint_separation(x):
        """(x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0"""
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                xi = x[3*i]
                yi = x[3*i+1]
                ri = x[3*i+2]
                xj = x[3*j]
                yj = x[3*j+1]
                rj = x[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                cons.append(dist_sq - sum_r**2)
        return np.array(cons)
    
    def constraint_radii_positive(x):
        """r_i >= 0.01"""
        return x[2::3][:N] - 0.01
    
    # Combine constraints
    def constraints_combined(x):
        c1 = constraint_perimeter(x)
        c2 = constraint_box(x)
        c3 = constraint_separation(x)
        c4 = constraint_radii_positive(x)
        return np.concatenate([[c1], c2, c3, c4])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0))  # x_i
        bounds.append((1e-4, 1.0))  # y_i
        bounds.append((0.01, 0.5))  # r_i
    bounds.append((0.1, 1.5))  # W
    bounds.append((0.1, 1.5))  # H
    
    # Try optimization
    best_result = None
    best_obj = np.inf
    
    # First, try SLSQP
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': constraint_perimeter},
                {'type': 'ineq', 'fun': constraint_box},
                {'type': 'ineq', 'fun': constraint_separation},
                {'type': 'ineq', 'fun': constraint_radii_positive}
            ],
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success or result.fun < best_obj:
            best_result = result
            best_obj = result.fun
        
        print(f"SLSQP: success={result.success}, obj={result.fun:.6f}", file=sys.stderr)
        
    except Exception as e:
        print(f"SLSQP failed: {e}", file=sys.stderr)
    
    # If SLSQP didn't work well, try trust-constr
    if best_obj > -0.5:  # If sum of radii is less than 0.5, try another method
        try:
            from scipy.optimize import LinearConstraint, NonlinearConstraint
            
            # Define constraints as arrays for NonlinearConstraint
            def con_perimeter(x):
                return 2.0 - (x[W_idx] + x[H_idx])
            
            def con_box(x):
                W = x[W_idx]
                H = x[H_idx]
                cons = []
                for i in range(N):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    ri = x[3*i+2]
                    cons.append(xi - ri)
                    cons.append(W - xi - ri)
                    cons.append(yi - ri)
                    cons.append(H - yi - ri)
                return np.array(cons)
            
            def con_sep(x):
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        xi = x[3*i]
                        yi = x[3*i+1]
                        ri = x[3*i+2]
                        xj = x[3*j]
                        yj = x[3*j+1]
                        rj = x[3*j+2]
                        dist_sq = (xi - xj)**2 + (yi - yj)**2
                        sum_r = ri + rj
                        cons.append(dist_sq - sum_r**2)
                return np.array(cons)
            
            def con_r(x):
                return x[2::3][:N] - 0.01
            
            # Use trust-constr
            cons = [
                NonlinearConstraint(con_perimeter, 0, np.inf),
                NonlinearConstraint(con_box, 0, np.inf),
                NonlinearConstraint(con_sep, 0, np.inf),
                NonlinearConstraint(con_r, 0, np.inf)
            ]
            
            result2 = minimize(
                objective,
                x0,
                method='trust-constr',
                jac=gradient,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'gtol': 1e-6}
            )
            
            if result2.success or result2.fun < best_obj:
                best_result = result2
                best_obj = result2.fun
            
            print(f"trust-constr: success={result2.success}, obj={result2.fun:.6f}", file=sys.stderr)
            
        except Exception as e:
            print(f"trust-constr failed: {e}", file=sys.stderr)
    
    # Extract the best result
    if best_result is not None and best_result.fun < np.inf:
        x_best = best_result.x
    else:
        x_best = x0
    
    # Extract circles
    circles = []
    for i in range(N):
        x = x_best[3*i]
        y = x_best[3*i+1]
        r = x_best[3*i+2]
        circles.append([x, y, r])
    
    circles = np.array(circles)
    
    # Verify and fix any issues
    # Ensure all radii are positive
    circles[:, 2] = np.maximum(circles[:, 2], 0.01)
    
    # Check bounding box
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    W = max_x - min_x
    H = max_y - min_y
    perimeter = 2 * (W + H)
    
    # If perimeter > 4, scale down
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        # Scale positions and radii
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        for i in range(N):
            circles[i, 0] = center_x + (circles[i, 0] - center_x) * scale
            circles[i, 1] = center_y + (circles[i, 1] - center_y) * scale
            circles[i, 2] = circles[i, 2] * scale
    
    return circles
# EVOLVE-BLOCK-END
