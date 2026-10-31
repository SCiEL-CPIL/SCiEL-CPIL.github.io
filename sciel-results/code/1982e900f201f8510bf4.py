"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_result = None
    
    # Parameters for optimization
    beta = 50.0  # Temperature for log-sum-exp
    
    # Define objective function
    def objective(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Calculate violations
        # Box constraints:
        # x_i >= r_i  => v = r_i - x_i
        # x_i <= 1-r_i => v = r_i - (1-x_i)
        # y_i >= r_i  => v = r_i - y_i
        # y_i <= 1-r_i => v = r_i - (1-y_i)
        v_box = np.concatenate([
            r - x,
            r - (1 - x),
            r - y,
            r - (1 - y)
        ])
        
        # Overlap constraints:
        # d_ij >= r_i + r_j => v = r_i + r_j - d_ij
        v_overlap = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                v_overlap.append(r[i] + r[j] - d)
        
        v_all = np.concatenate([v_box, v_overlap])
        
        # Log-sum-exp penalty
        # To avoid overflow, subtract max
        v_max = np.max(v_all)
        if v_max > 0:
            penalty = (v_max + np.log(np.sum(np.exp(v_all - v_max)))) / beta
        else:
            # If all violations are negative, penalty is small
            # log(sum(exp(v))) / beta
            # Since v are negative, exp(v) < 1
            # We can just compute it directly if values are not too large negative
            # But for safety, use the same formula
            penalty = (v_max + np.log(np.sum(np.exp(v_all - v_max)))) / beta
            
        return -sum_r + penalty

    # Define bounds
    # r > 0, x,y in [0,1]
    # We can tighten bounds slightly to help
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-4, 0.5)) # r

    # Restart loop
    while time.time() < deadline:
        # Initial guess
        # Random initialization
        np.random.seed(int(time.time() * 1000) % 10000) # Change seed each time?
        # Better: use a fixed seed for reproducibility in testing, but here we want diversity
        # Let's use random seed based on iteration count or just random
        
        # Grid initialization is often better
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Fill grid
        for i in range(N):
            col = i % side
            row = i // side
            x0[i] = (col + 0.5) / side
            y0[i] = (row + 0.5) / side
            r0[i] = 0.5 / side * 0.9 # Slightly smaller to start
            
        # Add jitter
        x0 += np.random.uniform(-0.05, 0.05, N)
        y0 += np.random.uniform(-0.05, 0.05, N)
        r0 += np.random.uniform(-0.01, 0.01, N)
        
        # Clip to valid range
        x0 = np.clip(x0, 0.0, 1.0)
        y0 = np.clip(y0, 0.0, 1.0)
        r0 = np.clip(r0, 1e-4, 0.5)
        
        # Ensure initial config is roughly valid or at least not terrible
        # Interleave x, y, r
        z0 = np.zeros(3 * N)
        z0[0::3] = x0
        z0[1::3] = y0
        z0[2::3] = r0
        
        # Optimize
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6})
        except Exception:
            continue
            
        # Extract result
        x = res.x[0::3]
        y = res.x[1::3]
        r = res.x[2::3]
        
        # Validate
        valid = True
        # Check bounds
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            valid = False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-5: # Small tolerance
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x, y, r))
                
    # If no valid result found, return the grid (seed behavior)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
