"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing the sum of radii.
    Uses a coverage-based formulation with differentiable soft-min operators
    to enforce that a grid of probe points is covered by the union of disks.
    This helps distribute circles evenly and avoid local minima.
    """
    n = N
    np.random.seed(42)
    
    # Define probe points: 5x5 grid
    grid_size = 5
    probe_x = np.linspace(0.1, 0.9, grid_size)
    probe_y = np.linspace(0.1, 0.9, grid_size)
    probes = np.array([(x, y) for x in probe_x for y in probe_y])
    num_probes = len(probes)
    
    # Initial placement: grid-like
    init_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            init_centers.append([probe_x[i], probe_y[j]])
    init_centers.append([0.5, 0.5]) # 26th circle
    
    init_centers = np.array(init_centers)
    init_radii = np.full(n, 0.1)
    
    # Parameters: x, y, r
    x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], init_radii])
    
    def unpack(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        return x, y, r
    
    def penalized_objective(params, mu=100.0):
        x, y, r = unpack(params)
        
        # Base objective: minimize negative sum of radii
        obj = -np.sum(r)
        
        penalty = 0.0
        
        # Non-overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                overlap = max(0, r[i] + r[j] - dist)
                penalty += overlap**2
        
        # Boundary penalty
        for i in range(n):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if 1 - x[i] < r[i]:
                penalty += (r[i] - (1 - x[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if 1 - y[i] < r[i]:
                penalty += (r[i] - (1 - y[i]))**2
                
        # Coverage penalty
        # For each probe, if it is not covered (softmin > 0), penalize.
        t = 10.0
        for k in range(num_probes):
            px = probes[k, 0]
            py = probes[k, 1]
            dx = x - px
            dy = y - py
            dists = np.sqrt(dx*dx + dy*dy + 1e-12)
            z = dists - r
            
            # Soft min
            z_max = np.max(z)
            exp_z = np.exp(-t * (z - z_max))
            sum_exp = np.sum(exp_z)
            soft_min_val = z_max - (1.0/t) * np.log(sum_exp)
            
            # If soft_min_val > 0, it's uncovered
            if soft_min_val > 0:
                penalty += soft_min_val**2
                
        return obj + mu * penalty
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-4, 0.5))
        
    best_params = x0.copy()
    best_val = penalized_objective(x0, mu=50.0)
    
    # Multi-stage optimization
    for mu in [10.0, 50.0, 100.0, 200.0, 500.0]:
        res = minimize(
            penalized_objective,
            best_params,
            args=(mu,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    # Refine
    for mu in [500.0, 1000.0]:
        res = minimize(
            penalized_objective,
            best_params,
            args=(mu,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
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
        
    for _ in range(10):
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
