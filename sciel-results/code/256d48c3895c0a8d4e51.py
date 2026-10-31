"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    best_result = None
    best_value = 0.0
    
    # We want to maximize sum of radii subject to:
    # 1. Rectangle perimeter <= 4: 2*(W+H) <= 4 => W+H <= 2
    # 2. Circles inside rectangle: cx_i - r_i >= 0, cx_i + r_i <= W, cy_i - r_i >= 0, cy_i + r_i <= H
    # 3. Non-overlapping: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
    
    # Variables: x = [W, H, cx_1..cx_N, cy_1..cy_N, r_1..r_N]
    # Total variables: 2 + 2*N + N = 3*N + 2 = 65
    
    def objective(x):
        W, H = x[0], x[1]
        cx = x[2:2+N]
        cy = x[2+N:2+2*N]
        r = x[2+2*N:2+3*N]
        return -np.sum(r)
    
    def constraints_check(x):
        W, H = x[0], x[1]
        cx = x[2:2+N]
        cy = x[2+N:2+2*N]
        r = x[2+2*N:2+3*N]
        
        # Perimeter constraint: W + H <= 2
        if W + H > 2 + 1e-8:
            return False
        
        # All circles must have positive radius
        if np.any(r <= 0):
            return False
        
        # Circles inside rectangle
        if np.any(cx - r < -1e-8) or np.any(cx + r > W + 1e-8):
            return False
        if np.any(cy - r < -1e-8) or np.any(cy + r > H + 1e-8):
            return False
        
        # Non-overlapping
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (cx[i]-cx[j])**2 + (cy[i]-cy[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2 - 1e-8:
                    return False
        
        return True
    
    # Try multiple random initializations
    random.seed(42)
    np.random.seed(42)
    
    num_restarts = 200
    attempts = 0
    
    while attempts < num_restarts:
        if time.time() > deadline - 5:
            break
        attempts += 1
        
        # Random aspect ratio W/H
        W = random.uniform(0.5, 1.5)
        H = 2.0 - W
        if H < 0.3:
            continue
        
        # Initialize circles in a grid-like pattern with some randomness
        side = int(np.ceil(np.sqrt(N)))
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            base_x = (col + 0.5) * W / side
            base_y = (row + 0.5) * H / side
            cx[i] = base_x + random.uniform(-0.1, 0.1) * W / side
            cy[i] = base_y + random.uniform(-0.1, 0.1) * H / side
            # Initial radius estimate
            r[i] = 0.5 * min(W, H) / side * 0.8
        
        # Ensure circles are within bounds
        for i in range(N):
            cx[i] = np.clip(cx[i], r[i], W - r[i])
            cy[i] = np.clip(cy[i], r[i], H - r[i])
        
        x0 = np.concatenate([[W, H], cx, cy, r])
        
        # Use SLSQP with constraints
        # Define constraints for SLSQP
        # We'll use penalty-based approach or direct constraints
        
        # Direct constraints are complex, so let's use a penalty method with scipy
        # Actually, let's try a different approach: use gradient-free optimization
        
        # Simple local search: try to increase radii and move circles
        current_x = x0.copy()
        current_valid = constraints_check(current_x)
        
        if not current_valid:
            # Try to fix by reducing radii
            for _ in range(10):
                r_idx = slice(2+2*N, 2+3*N)
                current_x[r_idx] *= 0.9
                if constraints_check(current_x):
                    break
            if not constraints_check(current_x):
                continue
        
        current_value = np.sum(current_x[2+2*N:2+3*N])
        
        # Local improvement loop
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try increasing a random circle's radius
            for _ in range(10):
                i = random.randint(0, N-1)
                r_idx = 2+2*N+i
                old_r = current_x[r_idx]
                new_r = old_r * 1.05
                trial = current_x.copy()
                trial[r_idx] = new_r
                
                # Check if we can move the circle to accommodate larger radius
                if constraints_check(trial):
                    current_x = trial
                    improved = True
                    break
                else:
                    # Try moving the circle
                    for dx in [0.01, -0.01, 0.02, -0.02, 0, 0]:
                        for dy in [0, 0, 0.01, -0.01, 0.02, -0.02]:
                            trial2 = trial.copy()
                            cx_idx = 2+i
                            cy_idx = 2+N+i
                            trial2[cx_idx] = trial[cx_idx] + dx
                            trial2[cy_idx] = trial[cy_idx] + dy
                            if constraints_check(trial2):
                                current_x = trial2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try moving circles to create space
            if not improved:
                i = random.randint(0, N-1)
                cx_idx = 2+i
                cy_idx = 2+N+i
                trial = current_x.copy()
                trial[cx_idx] += random.uniform(-0.05, 0.05)
                trial[cy_idx] += random.uniform(-0.05, 0.05)
                if constraints_check(trial):
                    current_x = trial
                    improved = True
            
            if not improved and step > 50:
                break
        
        if constraints_check(current_x):
            value = np.sum(current_x[2+2*N:2+3*N])
            if value > best_value:
                best_value = value
                best_result = current_x.copy()
    
    # Format result: array of (cx, cy, r)
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) * W / side
            cy = (row + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 1e-3
            cells.append([cx, cy, r])
        return np.array(cells)
    
    W, H = best_result[0], best_result[1]
    cx = best_result[2:2+N]
    cy = best_result[2+N:2+2*N]
    r = best_result[2+2*N:2+3*N]
    
    result = np.column_stack([cx, cy, r])
    return result

# EVOLVE-BLOCK-END
