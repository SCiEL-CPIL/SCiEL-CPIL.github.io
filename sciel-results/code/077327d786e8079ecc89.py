"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    
    def objective(params):
        """Objective function: negative of sum of radii (to minimize)."""
        W = params[0]
        H = params[1]
        r = params[2]
        centers = params[3:]
        
        # Perimeter constraint: 2*(W + H) = 4 => H = 2 - W
        # We enforce this in the constraint, but let's use it here for efficiency
        # Actually, let's just use the constraint to enforce W + H = 2
        
        # Check bounds: circles must be inside the rectangle
        for i in range(N):
            x = centers[2*i]
            y = centers[2*i+1]
            if x < r or x > W - r or y < r or y > H - r:
                # Penalty for being out of bounds
                penalty = 0
                if x < r:
                    penalty += (r - x)
                if x > W - r:
                    penalty += (x - (W - r))
                if y < r:
                    penalty += (r - y)
                if y > H - r:
                    penalty += (y - (H - r))
                if penalty > 1e-6:
                    return 1e6 * penalty
        
        # Check pairwise distances
        for i in range(N):
            xi = centers[2*i]
            yi = centers[2*i+1]
            for j in range(i+1, N):
                xj = centers[2*j]
                yj = centers[2*j+1]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < 2*r - 1e-6:
                    return 1e6 * (2*r - dist)
        
        # If all constraints satisfied, return negative sum of radii
        return -N * r
    
    def initial_guess(W, H):
        """Generate an initial guess for the optimizer."""
        # Use a grid pattern
        # Try different grid configurations
        best_guess = None
        best_min_dist = 0
        
        # 3x7 grid
        for rows, cols in [(3, 7), (7, 3), (4, 5), (5, 4)]:
            if rows * cols != N:
                # Adjust to get exactly N circles
                if rows * cols < N:
                    continue
                if rows * cols > N:
                    # Remove some circles
                    pass
            
            centers = []
            for r_idx in range(rows):
                for c_idx in range(cols):
                    x = (c_idx + 0.5) / cols * W
                    y = (r_idx + 0.5) / rows * H
                    centers.append((x, y))
            
            # Take first N
            centers = centers[:N]
            if len(centers) != N:
                continue
            
            # Calculate minimum distance
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                    min_dist = min(min_dist, dist)
            
            # Also check boundary distances
            for i in range(N):
                x, y = centers[i]
                min_dist = min(min_dist, x, W - x, y, H - y)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_guess = centers
        
        if best_guess is None:
            # Fallback: uniform random
            centers = []
            for _ in range(N):
                x = np.random.uniform(0.1, W - 0.1)
                y = np.random.uniform(0.1, H - 0.1)
                centers.append((x, y))
            best_guess = centers
        
        # Initial radius estimate
        r_init = best_min_dist / 2.0 * 0.9
        
        params = np.zeros(3 + 2*N)
        params[0] = W
        params[1] = H
        params[2] = r_init
        for i in range(N):
            params[3 + 2*i] = best_guess[i][0]
            params[3 + 2*i + 1] = best_guess[i][1]
        
        return params
    
    # Search over different aspect ratios
    best_value = 0.0
    best_solution = None
    
    # Perimeter 4 => W + H = 2
    # W in [0.5, 1.5]
    W_values = np.linspace(0.5, 1.5, 20)
    
    for W in W_values:
        H = 2.0 - W
        
        # Generate initial guess
        x0 = initial_guess(W, H)
        
        # Define constraints
        constraints = [
            # Perimeter constraint: W + H = 2
            {'type': 'eq', 'fun': lambda p: p[0] + p[1] - 2.0},
            # W > 0
            {'type': 'ineq', 'fun': lambda p: p[0] - 0.1},
            # H > 0
            {'type': 'ineq', 'fun': lambda p: p[1] - 0.1},
            # r > 0
            {'type': 'ineq', 'fun': lambda p: p[2] - 1e-6}
        ]
        
        # Bounds
        bounds = [
            (0.1, 1.9),  # W
            (0.1, 1.9),  # H
            (1e-6, 0.5)  # r
        ]
        for _ in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if result.success:
                W_opt = result.x[0]
                H_opt = result.x[1]
                r_opt = result.x[2]
                centers_opt = result.x[3:]
                
                # Verify solution
                valid = True
                for i in range(N):
                    x = centers_opt[2*i]
                    y = centers_opt[2*i+1]
                    if x < r_opt - 1e-4 or x > W_opt - r_opt + 1e-4 or y < r_opt - 1e-4 or y > H_opt - r_opt + 1e-4:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        xi = centers_opt[2*i]
                        yi = centers_opt[2*i+1]
                        for j in range(i+1, N):
                            xj = centers_opt[2*j]
                            yj = centers_opt[2*j+1]
                            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                            if dist < 2*r_opt - 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    value = N * r_opt
                    if value > best_value:
                        best_value = value
                        sol = np.zeros((N, 3))
                        sol[:, 0] = centers_opt[0::2]
                        sol[:, 1] = centers_opt[1::2]
                        sol[:, 2] = r_opt
                        best_solution = sol
        except Exception as e:
            pass
    
    # If optimization didn't find a good solution, use a fallback
    if best_solution is None:
        # Fallback: Simple 3x7 grid in 1x1
        coords = []
        for r in range(3):
            for c in range(7):
                coords.append(((c + 0.5) / 7.0, (r + 0.5) / 3.0))
        coords = np.array(coords)
        W = 1.0
        H = 1.0
        xs = coords[:, 0] * W
        ys = coords[:, 1] * H
        r_bound_x = np.min(np.minimum(xs, W - xs))
        r_bound_y = np.min(np.minimum(ys, H - ys))
        r_bound = min(r_bound_x, r_bound_y)
        r_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_pair = min(r_pair, dist / 2.0)
        r = min(r_bound, r_pair)
        sol = np.zeros((N, 3))
        sol[:, 0] = xs
        sol[:, 1] = ys
        sol[:, 2] = r
        best_solution = sol
    
    return best_solution
# EVOLVE-BLOCK-END
