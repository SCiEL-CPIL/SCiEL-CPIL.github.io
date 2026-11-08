"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses a local search / gradient-free optimization approach:
    1. Initialize with a reasonable grid-based layout.
    2. Iteratively adjust positions and radii to maximize sum of radii
       while maintaining non-overlap and boundary constraints.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = N
    
    # Initialize with a grid-like arrangement but with some randomness
    # to break symmetry and allow better packing
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add small random perturbation
            x += random.uniform(-0.02, 0.02)
            y += random.uniform(-0.02, 0.02)
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
            idx += 1
    
    # Ensure we have exactly n circles
    while len(cells) < n:
        cells.append([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 0.02])
    cells = cells[:n]
    
    def objective(params):
        """
        Minimize negative sum of radii (i.e., maximize sum of radii).
        params: array of shape (3*n,) = [x1, y1, r1, x2, y2, r2, ...]
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Penalty for circles going outside the unit square
        penalty = 0.0
        for i in range(n):
            if xs[i] < rs[i]:
                penalty += (rs[i] - xs[i]) ** 2 * 1000
            if xs[i] > 1.0 - rs[i]:
                penalty += (xs[i] - (1.0 - rs[i])) ** 2 * 1000
            if ys[i] < rs[i]:
                penalty += (rs[i] - ys[i]) ** 2 * 1000
            if ys[i] > 1.0 - rs[i]:
                penalty += (ys[i] - (1.0 - rs[i])) ** 2 * 1000
        
        # Penalty for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return -np.sum(rs) + penalty
    
    # Initial parameters
    init_params = np.array(cells).flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    best_result = init_params.copy()
    best_obj = objective(init_params)
    
    # Multiple restarts with different initial conditions
    for restart in range(5):
        if restart == 0:
            params = init_params.copy()
        else:
            # Perturb the best solution found so far
            params = best_result.copy()
            params += np.random.uniform(-0.05, 0.05, size=params.shape)
            # Clamp to valid ranges
            for i in range(n):
                params[3*i] = np.clip(params[3*i], 0.01, 0.99)
                params[3*i+1] = np.clip(params[3*i+1], 0.01, 0.99)
                params[3*i+2] = np.clip(params[3*i+2], 0.01, 0.5)
        
        result = minimize(
            objective, 
            params, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    
    # Extract final solution
    xs = best_result[0::3]
    ys = best_result[1::3]
    rs = best_result[2::3]
    
    # Post-processing: ensure all circles are valid
    # Clip radii and positions to be strictly inside
    for i in range(n):
        rs[i] = max(rs[i], 0.001)
        xs[i] = np.clip(xs[i], rs[i] + 1e-6, 1.0 - rs[i] - 1e-6)
        ys[i] = np.clip(ys[i], rs[i] + 1e-6, 1.0 - rs[i] - 1e-6)
    
    # Final check: if any circles overlap or are out of bounds, shrink them
    # to make the solution valid
    for _ in range(100):
        valid = True
        # Check boundaries
        for i in range(n):
            if xs[i] < rs[i] - 1e-6 or xs[i] > 1.0 - rs[i] + 1e-6:
                valid = False
                rs[i] = min(rs[i], xs[i] - 1e-6, 1.0 - xs[i] - 1e-6)
                if rs[i] < 0.001:
                    rs[i] = 0.001
                    xs[i] = np.clip(xs[i], rs[i] + 1e-6, 1.0 - rs[i] - 1e-6)
            if ys[i] < rs[i] - 1e-6 or ys[i] > 1.0 - rs[i] + 1e-6:
                valid = False
                rs[i] = min(rs[i], ys[i] - 1e-6, 1.0 - ys[i] - 1e-6)
                if rs[i] < 0.001:
                    rs[i] = 0.001
                    ys[i] = np.clip(ys[i], rs[i] + 1e-6, 1.0 - ys[i] - 1e-6)
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist - 1e-6:
                    valid = False
                    # Shrink both circles proportionally
                    overlap = min_dist - dist
                    total_r = rs[i] + rs[j]
                    if total_r > 0:
                        rs[i] -= overlap * (rs[i] / total_r) * 0.5
                        rs[j] -= overlap * (rs[j] / total_r) * 0.5
                        rs[i] = max(rs[i], 0.001)
                        rs[j] = max(rs[j], 0.001)
        
        if valid:
            break
    
    result = np.column_stack([xs, ys, rs])
    return result

# EVOLVE-BLOCK-END
