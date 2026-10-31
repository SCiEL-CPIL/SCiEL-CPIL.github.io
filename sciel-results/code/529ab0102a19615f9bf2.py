"""Rewritten program for circle packing in unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_result = None
    
    # Number of restarts. We'll adjust this if time is tight.
    n_restarts = 50
    
    for k in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Radii: start small, e.g., uniform [0.02, 0.1]
        # Positions: uniform [0.1, 0.9]
        r = np.random.uniform(0.02, 0.08, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        
        # Flatten to 1D vector for optimizer
        z0 = np.concatenate([x, y, r])
        
        # Objective function to minimize: -sum(r) + penalty
        def objective(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:]
            
            # Ensure radii are positive
            r = np.maximum(r, 0.001)
            
            sum_r = np.sum(r)
            
            # Boundary violations
            # x < r, x > 1-r, y < r, y > 1-r
            v_bound = 0.0
            v_bound += np.sum(np.maximum(0, r - x)**2)
            v_bound += np.sum(np.maximum(0, r - (1-x))**2)
            v_bound += np.sum(np.maximum(0, r - y)**2)
            v_bound += np.sum(np.maximum(0, r - (1-y))**2)
            
            # Collision violations
            # d_ij < r_i + r_j
            v_coll = 0.0
            # Vectorized collision check
            # dx = x[:, None] - x[None, :]
            # dy = y[:, None] - y[None, :]
            # d = sqrt(dx^2 + dy^2)
            # dist_matrix = np.sqrt(dx**2 + dy**2)
            # rad_sum = r[:, None] + r[None, :]
            # violations = np.maximum(0, rad_sum - dist_matrix)
            # v_coll = np.sum(violations**2)
            
            # Optimization for speed: use loops for N=26 is fast enough
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    req = r[i] + r[j]
                    if d < req:
                        v_coll += (req - d)**2
            
            penalty_weight = 100.0
            return -sum_r + penalty_weight * (v_bound + v_coll)
        
        # Run optimizer
        # SLSQP is good for constrained problems but we use penalty.
        # L-BFGS-B is faster but might struggle with non-smooth penalties if not careful.
        # Let's use L-BFGS-B with finite differences or analytic gradients?
        # Analytic gradients are complex. Finite differences for 78 variables is okay.
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success or res.fun < objective(z0):
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                # Post-processing: Ensure validity
                # If there are violations, shrink radii or move circles.
                # Simple check:
                valid = True
                for i in range(N):
                    if x_opt[i] < r_opt[i] or x_opt[i] > 1-r_opt[i] or \
                       y_opt[i] < r_opt[i] or y_opt[i] > 1-r_opt[i]:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if d < r_opt[i] + r_opt[j]:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    current_sum = np.sum(r_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
                        
        except Exception as e:
            continue
            
    # Fallback to seed if no better solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
