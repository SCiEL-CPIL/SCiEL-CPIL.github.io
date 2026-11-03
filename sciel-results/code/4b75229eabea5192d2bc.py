"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 21

# EVOLVE-BLOCK-START

def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_val = 0.0
    best_result = None
    
    # Generate candidate rectangles with perimeter 4
    # Perimeter 2(W+H)=4 => W+H=2
    # We'll try a range of aspect ratios
    # For 21 circles, a good starting aspect ratio is near 1:1 or slightly elongated
    # Let's parameterize W from 0.5 to 1.5 (since W+H=2, H=2-W)
    # W=1.0, H=1.0 is square
    # W=1.2, H=0.8 is 3:2
    # W=0.8, H=1.2 is 2:3
    
    # We'll use a local optimization approach:
    # Start with a grid-like packing, then optimize positions and radii
    
    # Strategy: 
    # 1. Try several initial configurations (different aspect ratios, different grid arrangements)
    # 2. For each, run a local optimization to maximize sum of radii subject to:
    #    - Circles are inside the rectangle [0,W] x [0,H]
    #    - Circles don't overlap: dist(i,j) >= r_i + r_j
    #    - All radii > 0
    # 3. Return the best configuration found
    
    def optimize_packing(W, H, initial_positions, initial_radii, max_time=30.0):
        """
        Optimize circle packing in a W x H rectangle.
        Returns (value, positions, radii) where value = sum of radii.
        """
        n = len(initial_radii)
        pos = np.array(initial_positions, dtype=float)  # (n, 2)
        rad = np.array(initial_radii, dtype=float)  # (n,)
        
        # Use gradient-free optimization: Nelder-Mead or a custom approach
        # For 21 circles, we have 42+21=63 variables, which is a lot for Nelder-Mead
        # Let's use a simpler approach: iterative local optimization
        
        # We'll use a simple gradient ascent with projected gradient
        # Objective: maximize sum(rad)
        # Constraints: 
        #   1. pos[i][0] >= rad[i], pos[i][0] <= W - rad[i]
        #   2. pos[i][1] >= rad[i], pos[i][1] <= H - rad[i]
        #   3. ||pos[i] - pos[j]|| >= rad[i] + rad[j] for all i < j
        
        # Let's use scipy's minimize with SLSQP
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback: return initial configuration
            val = np.sum(rad)
            return val, pos.copy(), rad.copy()
        
        def objective(x):
            """Negative sum of radii (we minimize negative to maximize)"""
            pos_x = x[:2*n].reshape(n, 2)
            rad_x = x[2*n:]
            # Penalty for constraint violations
            penalty = 0.0
            for i in range(n):
                # Rectangle bounds
                if pos_x[i][0] < rad_x[i]:
                    penalty += (rad_x[i] - pos_x[i][0]) ** 2
                if pos_x[i][0] > W - rad_x[i]:
                    penalty += (pos_x[i][0] - (W - rad_x[i])) ** 2
                if pos_x[i][1] < rad_x[i]:
                    penalty += (rad_x[i] - pos_x[i][1]) ** 2
                if pos_x[i][1] > H - rad_x[i]:
                    penalty += (pos_x[i][1] - (H - rad_x[i])) ** 2
            # Overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos_x[i] - pos_x[j])
                    min_d = rad_x[i] + rad_x[j]
                    if d < min_d:
                        penalty += (min_d - d) ** 2
            return -np.sum(rad_x) + 10.0 * penalty
        
        x0 = np.concatenate([initial_positions, initial_radii])
        
        # Run optimization with time limit
        t_opt_start = time.time()
        try:
            res = minimize(objective, x0, method='SLSQP', 
                          options={'maxiter': 200, 'ftol': 1e-8})
            t_opt_elapsed = time.time() - t_opt_start
        except Exception:
            return np.sum(rad), pos.copy(), rad.copy()
        
        if time.time() - t_start > deadline:
            return np.sum(rad), pos.copy(), rad.copy()
        
        x_opt = res.x
        pos_opt = x_opt[:2*n].reshape(n, 2)
        rad_opt = x_opt[2*n:]
        
        # Check validity and compute value
        valid = True
        for i in range(n):
            if rad_opt[i] <= 0:
                valid = False
                break
            if pos_opt[i][0] < rad_opt[i] - 1e-9 or pos_opt[i][0] > W - rad_opt[i] + 1e-9:
                valid = False
                break
            if pos_opt[i][1] < rad_opt[i] - 1e-9 or pos_opt[i][1] > H - rad_opt[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                    if d < rad_opt[i] + rad_opt[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(rad_opt)
            return val, pos_opt, rad_opt
        else:
            # Return initial if optimization failed
            val = np.sum(rad)
            return val, pos.copy(), rad.copy()
    
    # Generate initial configurations
    # Configuration 1: 5x5 grid (using 21 of 25 cells) in a square
    configs = []
    
    # Square 1x1
    W1, H1 = 1.0, 1.0
    side1 = 5
    cells1 = [((i + 0.5) / side1, (j + 0.5) / side1, 0.5 / side1 - 0.01) 
              for i in range(side1) for j in range(side1)]
    configs.append((W1, H1, [c[:2] for c in cells1[:N]], [c[2] for c in cells1[:N]]))
    
    # Rectangle 1.2 x 0.8 (3:2 aspect)
    W2, H2 = 1.2, 0.8
    # Use a grid that fits better
    nx2, ny2 = 5, 4  # 20 cells, add 1 more
    cells2 = []
    for i in range(nx2):
        for j in range(ny2):
            cells2.append(((i + 0.5) / nx2 * W2, (j + 0.5) / ny2 * H2, 
                          min(0.5 / nx2 * W2, 0.5 / ny2 * H2) - 0.01))
    # Add one more circle in the center
    cells2.append((W2/2, H2/2, min(W2, H2)/6 - 0.01))
    configs.append((W2, H2, [c[:2] for c in cells2[:N]], [c[2] for c in cells2[:N]]))
    
    # Rectangle 0.8 x 1.2 (2:3 aspect)
    W3, H3 = 0.8, 1.2
    nx3, ny3 = 4, 5  # 20 cells, add 1 more
    cells3 = []
    for i in range(nx3):
        for j in range(ny3):
            cells3.append(((i + 0.5) / nx3 * W3, (j + 0.5) / ny3 * H3, 
                          min(0.5 / nx3 * W3, 0.5 / ny3 * H3) - 0.01))
    cells3.append((W3/2, H3/2, min(W3, H3)/6 - 0.01))
    configs.append((W3, H3, [c[:2] for c in cells3[:N]], [c[2] for c in cells3[:N]]))
    
    # Hexagonal-like packing in square
    W4, H4 = 1.0, 1.0
    # Create a hexagonal pattern
    r4 = 0.18
    rows = 4
    cols = 5
    cells4 = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) * 2 * r4 * 0.95
            y = (row + 0.5) * 2 * r4 * 0.85
            if row % 2 == 1:
                x += r4 * 0.95
            cells4.append((x, y, r4 - 0.01))
    # Take first 21
    configs.append((W4, H4, [c[:2] for c in cells4[:N]], [c[2] for c in cells4[:N]]))
    
    # Try each configuration
    for W, H, init_pos, init_rad in configs:
        if time.time() - t_start > deadline - 5.0:
            break
        
        val, pos, rad = optimize_packing(W, H, init_pos, init_rad)
        
        # Scale positions to fit in [0, W] x [0, H] and ensure validity
        # Actually, the optimizer should have handled this
        # But let's clamp to be safe
        for i in range(N):
            rad[i] = max(rad[i], 0.01)
            pos[i][0] = np.clip(pos[i][0], rad[i], W - rad[i])
            pos[i][1] = np.clip(pos[i][1], rad[i], H - rad[i])
        
        # Check if valid
        valid = True
        for i in range(N):
            if pos[i][0] < rad[i] - 1e-9 or pos[i][0] > W - rad[i] + 1e-9:
                valid = False
                break
            if pos[i][1] < rad[i] - 1e-9 or pos[i][1] > H - rad[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            result = np.zeros((N, 3))
            for i in range(N):
                result[i] = [pos[i][0], pos[i][1], rad[i]]
            val = np.sum(rad)
            if val > best_val:
                best_val = val
                best_result = result
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_val = np.sum(best_result[:, 2])
    
    return best_result

# EVOLVE-BLOCK-END
