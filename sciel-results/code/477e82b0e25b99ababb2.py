"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square maximizing the sum of radii.
    Uses a local search / optimization approach starting from a grid-based
    initialization, then iteratively improving via gradient-free optimization
    on the combined objective (sum of radii) with non-overlap and boundary constraints.
    """
    
    # We'll use a penalty-based optimization approach.
    # Variables: x[0:N], y[0:N], r[0:N] -> total 3*N variables
    # Objective: maximize sum(r) -> minimize -sum(r)
    # Constraints:
    #   1. For all i < j: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #   2. 0 <= x_i <= 1, 0 <= y_i <= 1
    #   3. r_i >= 0
    #   4. x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    
    n = N
    dim = 3 * n
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(n)))
    x_init = np.zeros(n)
    y_init = np.zeros(n)
    r_init = np.zeros(n)
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                x_init[count] = (i + 0.5) / side
                y_init[count] = (j + 0.5) / side
                r_init[count] = 0.5 / side - 1e-3
                count += 1
    
    x0 = np.concatenate([x_init, y_init, r_init])
    
    # Penalty function
    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Base objective: negative sum of radii
        val = -np.sum(r)
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2:
                    # Penalty proportional to violation
                    violation = min_dist - np.sqrt(dist_sq + 1e-12)
                    penalty += 100.0 * violation**2
        
        # Penalty for boundary violations
        for i in range(n):
            if x[i] - r[i] < 0:
                penalty += 100.0 * (x[i] - r[i])**2
            if x[i] + r[i] > 1:
                penalty += 100.0 * (x[i] + r[i] - 1)**2
            if y[i] - r[i] < 0:
                penalty += 100.0 * (y[i] - r[i])**2
            if y[i] + r[i] > 1:
                penalty += 100.0 * (y[i] + r[i] - 1)**2
            if r[i] < 0:
                penalty += 100.0 * r[i]**2
        
        return val + penalty
    
    # Try multiple random restarts to find a good solution
    best_val = np.inf
    best_v = x0.copy()
    
    # First try from the grid initialization
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    if res.fun < best_val:
        best_val = res.fun
        best_v = res.x.copy()
    
    # Try a few random restarts
    np.random.seed(42)
    for trial in range(5):
        # Random initialization with small circles
        x_r = np.random.uniform(0.1, 0.9, n)
        y_r = np.random.uniform(0.1, 0.9, n)
        r_r = np.random.uniform(0.02, 0.08, n)
        v_r = np.concatenate([x_r, y_r, r_r])
        
        res = minimize(objective, v_r, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_v = res.x.copy()
    
    x_final = best_v[0:n]
    y_final = best_v[n:2*n]
    r_final = best_v[2*n:3*n]
    
    # Clamp to valid range and ensure non-overlap with small adjustments
    r_final = np.maximum(r_final, 1e-6)
    x_final = np.clip(x_final, r_final, 1 - r_final)
    y_final = np.clip(y_final, r_final, 1 - r_final)
    
    # Final overlap resolution: simple iterative push-apart
    for iteration in range(100):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x_final[i] - x_final[j]
                dy = y_final[i] - y_final[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r_final[i] + r_final[j]
                if dist < min_dist:
                    # Push apart
                    overlap = min_dist - dist
                    if dist < 1e-10:
                        dx = 0.01
                        dy = 0.01
                        dist = np.sqrt(dx*dx + dy*dy)
                    unit_x = dx / dist
                    unit_y = dy / dist
                    x_final[i] += unit_x * overlap * 0.5
                    y_final[i] += unit_y * overlap * 0.5
                    x_final[j] -= unit_x * overlap * 0.5
                    y_final[j] -= unit_y * overlap * 0.5
                    moved = True
        
        # Re-clamp to boundaries
        for i in range(n):
            if x_final[i] < r_final[i]:
                x_final[i] = r_final[i]
            if x_final[i] > 1 - r_final[i]:
                x_final[i] = 1 - r_final[i]
            if y_final[i] < r_final[i]:
                y_final[i] = r_final[i]
            if y_final[i] > 1 - r_final[i]:
                y_final[i] = 1 - r_final[i]
        
        if not moved:
            break
    
    # Verify and build result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = [x_final[i], y_final[i], r_final[i]]
    
    return result
# EVOLVE-BLOCK-END
