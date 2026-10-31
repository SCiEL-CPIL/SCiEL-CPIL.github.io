"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # We will try several random restarts
    num_restarts = 50  # Adjust based on time per restart
    
    rng = np.random.default_rng(42)
    
    def objective(params):
        # params: [x0, y0, r0, x1, y1, r1, ...]
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x in [r, 1-r], y in [r, 1-r]
        # Penalty if x < r or x > 1-r
        pen_x = np.sum(np.maximum(0, r - x)**2 + np.maximum(0, x + r - 1)**2)
        pen_y = np.sum(np.maximum(0, r - y)**2 + np.maximum(0, y + r - 1)**2)
        
        # Overlap penalties
        # d_ij >= r_i + r_j
        # d^2 >= (r_i + r_j)^2
        # Penalty: max(0, (r_i+r_j)^2 - d^2)^2 ? Or max(0, r_i+r_j-d)^2
        # Using distance is more geometrically intuitive but sqrt is expensive.
        # Let's use squared distance penalty for efficiency, scaled appropriately.
        # If d^2 < (r_i+r_j)^2, penalty is ((r_i+r_j)^2 - d^2)^2
        # This might be too steep or shallow. Let's stick to distance based penalty for robustness, 
        # but N=26 means 325 pairs. 325 sqrts per evaluation is fine.
        
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    pen_overlap += (min_d - d)**2
        
        penalty = pen_x + pen_y + pen_overlap
        return -sum_r + 100.0 * penalty
    
    # Initial guess strategy: Random positions, random small radii
    # Or a jittered grid
    
    for k in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize
        # Random positions in [0.1, 0.9]
        x0 = rng.uniform(0.1, 0.9, N)
        y0 = rng.uniform(0.1, 0.9, N)
        r0 = rng.uniform(0.01, 0.05, N) # Small initial radii
        
        # Construct flat array
        init_params = np.zeros(3*N)
        init_params[0::3] = x0
        init_params[1::3] = y0
        init_params[2::3] = r0
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 0.5)) # r
            
        # Run optimization
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Extract solution
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            # Check validity and calculate score
            # The objective function returns -sum_r + penalty.
            # If penalty is 0, score is sum_r.
            # We should verify constraints are satisfied.
            
            valid = True
            # Check boundaries
            for i in range(N):
                if x_opt[i] < r_opt[i] - 1e-9 or x_opt[i] > 1 - r_opt[i] + 1e-9:
                    valid = False
                if y_opt[i] < r_opt[i] - 1e-9 or y_opt[i] > 1 - r_opt[i] + 1e-9:
                    valid = False
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    if d < r_opt[i] + r_opt[j] - 1e-9:
                        valid = False
            
            if valid:
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_solution = np.column_stack((x_opt, y_opt, r_opt))
            else:
                # If invalid, we might still keep it if the penalty was very small, 
                # but the task requires disjoint interiors. 
                # We can try to "fix" by slightly shrinking radii? 
                # Or just discard. Discarding is safer for validity.
                pass
                
        except Exception as e:
            pass

    if best_solution is not None:
        return best_solution
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
