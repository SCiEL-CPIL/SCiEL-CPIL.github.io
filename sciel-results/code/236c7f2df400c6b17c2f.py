"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint

N = 26

# EVOLVE-BLOCK-START

def _pack_with_slsqp(x0, y0, r0, max_time=110.0):
    """
    Pack N circles using SLSQP with hard constraints.
    Variables: x[0:N], y[0:N], r[0:N]
    Constraints:
      - 0 <= x_i <= 1
      - 0 <= y_i <= 1
      - r_i >= 0
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
      - r_i + r_j <= sqrt((x_i-x_j)^2 + (y_i-y_j)^2) for all i<j (non-overlap)
      - x_i - r_i >= 0, y_i - r_i >= 0, 1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
    Maximize sum(r_i) => minimize -sum(r_i)
    """
    n = N
    dim = 3 * n
    
    def objective(z):
        r = z[2*n:]
        return -np.sum(r)
    
    # Build constraints
    constraints = []
    
    # Boundary constraints: each circle must be inside [0,1]^2
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    # r_i >= 0
    
    # We'll use inequality constraints g(z) >= 0
    
    def boundary_constraints(z):
        x = z[:n]
        y = z[n:2*n]
        r = z[2*n:]
        return np.concatenate([
            x - r,          # x_i - r_i >= 0
            1 - x - r,      # 1 - x_i - r_i >= 0
            y - r,          # y_i - r_i >= 0
            1 - y - r,      # 1 - y_i - r_i >= 0
            r               # r_i >= 0
        ])
    
    constraints.append(NonlinearConstraint(boundary_constraints, 0, np.inf))
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    def non_overlap_constraints(z):
        x = z[:n]
        y = z[n:2*n]
        r = z[2*n:]
        vals = []
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                vals.append(dx*dx + dy*dy - dr*dr)
        return np.array(vals)
    
    constraints.append(NonlinearConstraint(non_overlap_constraints, 0, np.inf))
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
    for i in range(n):
        bounds.append((0, 1))  # y_i
    for i in range(n):
        bounds.append((0, 0.5))  # r_i (max radius is 0.5)
    
    # Initial guess
    z0 = np.concatenate([x0, y0, r0])
    
    start_time = time.time()
    
    try:
        result = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success and time.time() - start_time < max_time:
            return result.x
        else:
            return z0
    except Exception:
        return z0


def construct():
    n = N
    deadline = time.time() + 115.0  # safety margin
    
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(n)))
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    r0 = np.zeros(n)
    
    for i in range(n):
        row = i // side
        col = i % side
        x0[i] = (col + 0.5) / side
        y0[i] = (row + 0.5) / side
        r0[i] = 0.5 / side - 0.01
    
    best_z = np.concatenate([x0, y0, r0])
    best_val = -np.sum(r0)
    
    # Try a few different initializations
    initializations = []
    
    # 1. Grid
    initializations.append(np.concatenate([x0, y0, r0]))
    
    # 2. Random perturbation of grid
    rng = np.random.default_rng(42)
    for trial in range(3):
        x_trial = x0 + rng.uniform(-0.05, 0.05, n)
        y_trial = y0 + rng.uniform(-0.05, 0.05, n)
        r_trial = np.full(n, 0.5 / side - 0.02)
        x_trial = np.clip(x_trial, 0, 1)
        y_trial = np.clip(y_trial, 0, 1)
        initializations.append(np.concatenate([x_trial, y_trial, r_trial]))
    
    # 3. Smaller circles, more spread out
    r_small = np.full(n, 0.1)
    x_small = rng.uniform(0.15, 0.85, n)
    y_small = rng.uniform(0.15, 0.85, n)
    initializations.append(np.concatenate([x_small, y_small, r_small]))
    
    for z_init in initializations:
        if time.time() > deadline - 30:
            break
        
        z_result = _pack_with_slsqp(
            z_init[:n], 
            z_init[n:2*n], 
            z_init[2*n:],
            max_time=100
        )
        
        # Check validity
        x = z_result[:n]
        y = z_result[n:2*n]
        r = z_result[2*n:]
        
        # Validate
        valid = True
        for i in range(n):
            if x[i] < -1e-9 or x[i] > 1 + 1e-9 or y[i] < -1e-9 or y[i] > 1 + 1e-9 or r[i] < -1e-9:
                valid = False
                break
            if x[i] - r[i] < -1e-7 or 1 - x[i] - r[i] < -1e-7 or y[i] - r[i] < -1e-7 or 1 - y[i] - r[i] < -1e-7:
                valid = False
                break
        
        if valid:
            # Check non-overlap
            for i in range(n):
                for j in range(i+1, n):
                    dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    sum_r = r[i] + r[j]
                    if dist2 < sum_r**2 - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(r)
            if val > -best_val:  # best_val is negative sum
                best_val = -val
                best_z = z_result.copy()
    
    # Final result
    x = best_z[:n]
    y = best_z[n:2*n]
    r = best_z[2*n:]
    
    # Clip to ensure validity
    r = np.maximum(r, 0)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    result = np.zeros((n, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result

# EVOLVE-BLOCK-END
