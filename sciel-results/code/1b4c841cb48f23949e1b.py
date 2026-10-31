"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing the sum of radii.
    Uses a vectorized coverage-based formulation with differentiable soft-min operators.
    """
    n = N
    
    # Initial placement: 5x5 grid + 1 in a hole
    init_centers = []
    for i in range(5):
        for j in range(5):
            x = 0.1 + j * 0.2
            y = 0.1 + i * 0.2
            init_centers.append([x, y])
    # Add 26th at a hole (0.2, 0.2)
    init_centers.append([0.2, 0.2])
    init_centers = np.array(init_centers)
    init_radii = np.full(n, 0.05)
    
    x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], init_radii])
    
    def unpack(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        return x, y, r
    
    # Define probe points: 10x10 grid for better spread enforcement
    grid_size = 10
    probe_x = np.linspace(0.05, 0.95, grid_size)
    probe_y = np.linspace(0.05, 0.95, grid_size)
    probes = np.array([(x, y) for x in probe_x for y in probe_y])
    
    def penalized_objective(params, mu=100.0, t=20.0):
        x, y, r = unpack(params)
        
        obj = -np.sum(r)
        penalty = 0.0
        
        # Vectorized Non-overlap penalty
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_mat = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        overlap_mat = np.maximum(0, r_sum - dist_mat)
        iu = np.triu_indices(n, k=1)
        penalty += np.sum(overlap_mat[iu]**2)
        
        # Vectorized Boundary penalty
        pen_x_left = np.maximum(0, r - x)**2
        pen_x_right = np.maximum(0, r - (1 - x))**2
        pen_y_bottom = np.maximum(0, r - y)**2
        pen_y_top = np.maximum(0, r - (1 - y))**2
        penalty += np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # Vectorized Coverage penalty
        dx_p = probes[:, 0:1] - x[np.newaxis, :]
        dy_p = probes[:, 1:2] - y[np.newaxis, :]
        dists = np.sqrt(dx_p**2 + dy_p**2 + 1e-12)
        z = dists - r[np.newaxis, :]
        
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(-t * (z - z_max))
        sum_exp = np.sum(exp_z, axis=1)
        soft_min_val = (z_max - (1.0/t) * np.log(sum_exp + 1e-12)).flatten()
        
        penalty += np.sum(np.maximum(0, soft_min_val)**2)
        
        return obj + mu * penalty
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-4, 0.5))
        
    best_params = x0.copy()
    best_val = penalized_objective(x0, mu=50.0, t=10.0)
    
    # Multi-stage optimization
    stages = [
        {'mu': 10.0, 't': 5.0, 'maxiter': 200},
        {'mu': 50.0, 't': 10.0, 'maxiter': 300},
        {'mu': 100.0, 't': 20.0, 'maxiter': 500},
        {'mu': 500.0, 't': 50.0, 'maxiter': 500},
        {'mu': 1000.0, 't': 100.0, 'maxiter': 300},
    ]
    
    for stage in stages:
        res = minimize(
            penalized_objective,
            best_params,
            args=(stage['mu'], stage['t']),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': stage['maxiter'], 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    x, y, r = unpack(best_params)
    
    # Cleanup
    r = np.maximum(r, 1e-6)
    for i in range(n):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
        
    for _ in range(20):
        adjusted = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                required = r[i] + r[j]
                if dist < required:
                    scale = dist / (required + 1e-12)
                    r[i] *= scale
                    r[j] *= scale
                    adjusted = True
        if not adjusted:
            break
            
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
