"""Rewritten program for circle packing in unit square; using NLP optimization."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses scipy.optimize.minimize with L-BFGS-B and soft penalty functions.
    """
    n = N
    
    # Objective function with soft penalties for constraints
    def soft_objective(z, lam=10.0):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Base objective: maximize sum of radii (minimize negative)
        obj = -np.sum(r)
        
        # Boundary penalties
        # x < r -> penalty (r - x)^2
        # x > 1-r -> penalty (x - (1-r))^2
        pen_x_low = np.sum(np.maximum(0, r - x)**2)
        pen_x_high = np.sum(np.maximum(0, x - (1 - r))**2)
        pen_y_low = np.sum(np.maximum(0, r - y)**2)
        pen_y_high = np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Overlap penalties
        # dist < ri + rj -> penalty (ri + rj - dist)^2
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        return obj + lam * (pen_x_low + pen_x_high + pen_y_low + pen_y_high + overlap_pen)
    
    # Bounds for L-BFGS-B
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0.001, 0.5)) # r
    
    best_score = -np.inf
    best_z = None
    
    np.random.seed(42)
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if restart == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(n)))
            z0 = np.zeros(3*n)
            for i in range(side):
                for j in range(side):
                    idx = i * side + j
                    if idx < n:
                        z0[3*idx] = (i + 0.5) / side
                        z0[3*idx + 1] = (j + 0.5) / side
                        z0[3*idx + 2] = 0.5 / side * 0.9
        elif restart == 1:
            # Hexagonal lattice initialization
            z0 = np.zeros(3*n)
            dx = 1.0 / np.sqrt(n)
            dy = dx * np.sqrt(3) / 2
            count = 0
            for row in range(int(1.0/dy) + 1):
                for col in range(int(1.0/dx) + 1):
                    if count >= n:
                        break
                    x_pos = col * dx + (row % 2) * dx / 2
                    y_pos = row * dy
                    if x_pos < 1 and y_pos < 1:
                        z0[3*count] = x_pos + 0.05
                        z0[3*count + 1] = y_pos + 0.05
                        z0[3*count + 2] = dx / 2 * 0.9
                        count += 1
        else:
            # Random initialization
            z0 = np.zeros(3*n)
            z0[0::3] = np.random.uniform(0.1, 0.9, n)
            z0[1::3] = np.random.uniform(0.1, 0.9, n)
            z0[2::3] = np.random.uniform(0.05, 0.1, n)
            
        # Optimize
        try:
            res = minimize(soft_objective, z0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10})
            z_opt = res.x
        except Exception:
            continue
        
        # Check validity of the solution (soft constraints might not be fully satisfied)
        x = z_opt[0::3]
        y = z_opt[1::3]
        r = z_opt[2::3]
        
        # Verify hard constraints
        valid = True
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            valid = False
        if np.any(r < 1e-5):
            valid = False
            
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist**2 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_z = z_opt.copy()
    
    if best_z is None:
        # Fallback to a simple valid configuration
        side = int(np.ceil(np.sqrt(n)))
        z0 = np.zeros(3*n)
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx < n:
                    z0[3*idx] = (i + 0.5) / side
                    z0[3*idx + 1] = (j + 0.5) / side
                    z0[3*idx + 2] = 0.5 / side * 0.9
        best_z = z0
        best_score = np.sum(best_z[2::3])
        
    return np.column_stack([best_z[0::3], best_z[1::3], best_z[2::3]])
# EVOLVE-BLOCK-END
