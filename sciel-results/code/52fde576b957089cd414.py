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
    
    # Define probe points: 10x10 grid for better coverage enforcement
    grid_size = 10
    probe_x = np.linspace(0.05, 0.95, grid_size)
    probe_y = np.linspace(0.05, 0.95, grid_size)
    probes = np.array([(x, y) for x in probe_x for y in probe_y])
    
    def unpack(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        return x, y, r
    
    def penalized_objective(params, mu_overlap, mu_cover, mu_bound):
        x, y, r = unpack(params)
        
        # Base objective: minimize negative sum of radii
        obj = -np.sum(r)
        
        # Non-overlap penalty (vectorized)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlaps = np.maximum(0, r_sum[mask] - dist[mask])
        pen_overlap = np.sum(overlaps**2)
        
        # Boundary penalty (vectorized)
        pen_bound = 0
        pen_bound += np.sum(np.maximum(0, r - x)**2)
        pen_bound += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_bound += np.sum(np.maximum(0, r - y)**2)
        pen_bound += np.sum(np.maximum(0, r - (1 - y))**2)
        
        # Coverage penalty (vectorized)
        dx_p = x[:, np.newaxis] - probes[np.newaxis, :, 0]
        dy_p = y[:, np.newaxis] - probes[np.newaxis, :, 1]
        dist_p = np.sqrt(dx_p*dx_p + dy_p*dy_p + 1e-12)
        adjusted_dist = dist_p - r[:, np.newaxis]
        
        t = 10.0
        z = adjusted_dist
        z_max = np.max(z, axis=0)
        exp_z = np.exp(-t * (z - z_max))
        sum_exp = np.sum(exp_z, axis=0)
        soft_min_val = z_max - (1.0/t) * np.log(sum_exp)
        
        uncovered = np.maximum(0, soft_min_val)
        pen_cover = np.sum(uncovered**2)
        
        return obj + mu_overlap * pen_overlap + mu_bound * pen_bound + mu_cover * pen_cover
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-4, 0.5))
        
    best_result = None
    best_score = -np.inf
    
    # Try multiple random restarts
    for seed in range(20):
        np.random.seed(seed)
        init_centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        init_radii = np.full(n, 0.05)
        x0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], init_radii])
        
        best_params = x0.copy()
        best_val = penalized_objective(x0, 5.0, 0.5, 5.0)
        
        # Multi-stage optimization
        stages = [
            (5.0, 0.5, 5.0, 300),
            (10.0, 1.0, 10.0, 300),
            (20.0, 2.0, 20.0, 300),
            (50.0, 5.0, 50.0, 500),
            (100.0, 10.0, 100.0, 500)
        ]
        
        for mu_o, mu_c, mu_b, maxiter in stages:
            res = minimize(
                penalized_objective,
                best_params,
                args=(mu_o, mu_c, mu_b),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6}
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
            
        # Resolve overlaps
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
        
        # Validate
        valid = True
        if np.any(r < 0) or np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            valid = False
        for i in range(n):
            if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                valid = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    
        if valid:
            sum_r = np.sum(r)
            if sum_r > best_score:
                best_score = sum_r
                best_result = result
                
    if best_result is None:
        # Fallback
        x, y, r = [], [], []
        grid = 5
        step = 1.0 / grid
        for i in range(grid):
            for j in range(grid):
                x.append(i*step + step/2)
                y.append(j*step + step/2)
                r.append(step/2 * 0.9)
        x.append(0.5)
        y.append(0.5)
        r.append(0.05)
        best_result = np.column_stack([x, y, r])
        
    return best_result

# EVOLVE-BLOCK-END
