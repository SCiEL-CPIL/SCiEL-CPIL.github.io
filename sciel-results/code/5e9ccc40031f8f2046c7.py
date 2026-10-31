"""Rewritten program for circle packing in unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 maximizing sum of radii.
    Returns array of shape (N, 3) with rows (x, y, radius).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Initial guess: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
    x0 = np.array(positions)
    r0 = np.full(N, 0.5 / side)
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    # Total variables: 3*N
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
    
    def unflatten(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r
    
    def objective(v):
        """Negative sum of radii (minimize negative to maximize sum)."""
        x, y, r = unflatten(v)
        return -np.sum(r)
    
    def constraints_func(v):
        """
        Returns list of constraint functions for SLSQP.
        Each constraint: g_i(v) >= 0
        """
        x, y, r = unflatten(v)
        constraints = []
        
        # Circle i must be inside unit square:
        # x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i] - v[3*i+2]
            })
            # x_i + r_i <= 1 => 1 - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
            })
            # y_i + r_i <= 1 => 1 - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]
            })
        
        # Non-overlap: for i < j, (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Equivalent: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                })
        
        # Radii must be positive
        for i in range(N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+2] - 1e-6
            })
        
        return constraints
    
    # Build initial point
    v0 = flatten(x0, np.array([p[1] for p in positions]), r0)
    
    # Bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((1e-6, 0.5))  # r_i
    
    # Run SLSQP optimization with multiple restarts
    best_obj = np.inf
    best_v = v0.copy()
    
    # Try a few different initial configurations
    initial_configs = []
    
    # Config 1: grid
    initial_configs.append(v0.copy())
    
    # Config 2: random perturbations of grid
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        v_rand = v0.copy()
        # Small random perturbation
        v_rand[0::3] += rng.uniform(-0.05, 0.05, N)  # x
        v_rand[1::3] += rng.uniform(-0.05, 0.05, N)  # y
        # Clip to valid range
        v_rand[0::3] = np.clip(v_rand[0::3], 0.01, 0.99)
        v_rand[1::3] = np.clip(v_rand[1::3], 0.01, 0.99)
        v_rand[2::3] = np.clip(v_rand[2::3], 1e-4, 0.45)
        initial_configs.append(v_rand)
    
    constraints = constraints_func(v0)
    
    for config_idx, v_init in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                v_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or (not result.success and np.sum(unflatten(result.x)[2]) > np.sum(unflatten(best_v)[2])):
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_v = result.x.copy()
                    
        except Exception as e:
            continue
    
    # Extract best result
    x, y, r = unflatten(best_v)
    
    # Ensure feasibility: clip radii and positions to satisfy constraints
    # For each circle, ensure it's inside the square
    for i in range(N):
        # Adjust radius to fit in square given position
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = min(r[i], max(max_r, 1e-6))
    
    # Final feasibility check and adjustment for overlaps
    # Do a simple iterative separation if any overlaps remain
    for iteration in range(50):
        if time.time() > deadline:
            break
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-10:
                    overlap_found = True
                    # Reduce the smaller radius
                    if r[i] < r[j]:
                        r[i] = max(min_dist - r[j], 1e-6)
                    else:
                        r[j] = max(min_dist - r[i], 1e-6)
                    # Re-check bounds
                    for k in [i, j]:
                        max_r = min(x[k], 1 - x[k], y[k], 1 - y[k])
                        r[k] = min(r[k], max(max_r, 1e-6))
        if not overlap_found:
            break
    
    result_array = np.column_stack([x, y, r])
    return result_array

# EVOLVE-BLOCK-END
