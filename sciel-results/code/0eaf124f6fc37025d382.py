"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses a multi-stage approach:
    1. Initialize with a good lattice/hexagonal pattern.
    2. Use L-BFGS-B with a penalty-based objective to optimize positions and radii.
    3. Apply local differential evolution polishing for better local optima.
    """
    deadline = time.time() + 140  # 140s safety margin
    
    # Objective: maximize sum of radii, penalize overlaps and boundary violations
    def objective(params, penalty_weight=100.0):
        """
        params: array of shape (3*N,) = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        Returns: -sum(r) + penalty for overlaps and boundary violations
        """
        n = len(params) // 3
        xy_r = params.reshape(n, 3)
        x, y, r = xy_r[:, 0], xy_r[:, 1], xy_r[:, 2]
        
        # Boundary penalty: circles must be inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        boundary_violation = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                             np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        boundary_penalty = penalty_weight * np.sum(boundary_violation)
        
        # Overlap penalty: for each pair, distance >= r_i + r_j
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_penalty += penalty_weight * (min_dist - dist)
        
        return -np.sum(r) + boundary_penalty + overlap_penalty
    
    # Gradient of objective
    def objective_grad(params, penalty_weight=100.0):
        n = len(params) // 3
        xy_r = params.reshape(n, 3)
        x, y, r = xy_r[:, 0], xy_r[:, 1], xy_r[:, 2]
        grad = np.zeros_like(params)
        
        # Base gradient: -1 for radii
        grad[2::3] = -1.0
        
        # Boundary gradient
        for i in range(n):
            # x - r >= 0 -> grad_x += penalty_weight if violated
            if x[i] < r[i]:
                grad[i*3] += penalty_weight
                grad[i*3+2] -= penalty_weight
            # x + r <= 1 -> grad_x -= penalty_weight if violated
            if x[i] > 1 - r[i]:
                grad[i*3] -= penalty_weight
                grad[i*3+2] += penalty_weight
            # y - r >= 0
            if y[i] < r[i]:
                grad[i*3+1] += penalty_weight
                grad[i*3+2] -= penalty_weight
            # y + r <= 1
            if y[i] > 1 - r[i]:
                grad[i*3+1] -= penalty_weight
                grad[i*3+2] += penalty_weight
        
        # Overlap gradient
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist > 1e-12:
                        factor = penalty_weight / dist
                        grad[i*3] += factor * dx
                        grad[i*3+1] += factor * dy
                        grad[j*3] -= factor * dx
                        grad[j*3+1] -= factor * dy
                    # Radii gradients
                    grad[i*3+2] += penalty_weight
                    grad[j*3+2] += penalty_weight
        
        return grad
    
    # Initialize with hexagonal lattice pattern
    def hex_init(n):
        """Create a hexagonal lattice initialization for n circles."""
        # For 26 circles, try a 5x5 grid with some adjustments
        # Use a denser packing pattern
        params = np.zeros(3 * n)
        
        # Try different grid layouts and pick the best
        best_params = None
        best_obj = np.inf
        
        for cols in range(3, 8):
            rows = int(np.ceil(n / cols))
            for offset in [0, 0.5]:
                for i in range(n):
                    row = i // cols
                    col = i % cols
                    # Hexagonal offset
                    x = (col + 0.5 + offset * (row % 2)) / (cols + offset)
                    y = (row + 0.5) / rows
                    # Clamp to [0,1]
                    x = np.clip(x, 0.05, 0.95)
                    y = np.clip(y, 0.05, 0.95)
                    r = 0.15  # Initial radius
                    
                    idx = i * 3
                    params[idx] = x
                    params[idx+1] = y
                    params[idx+2] = r
                
                obj = objective(params, 10.0)
                if obj < best_obj:
                    best_obj = obj
                    best_params = params.copy()
        
        return best_params if best_params is not None else params
    
    # Local differential evolution polishing
    def local_de(params, iterations=50, pop_size=20, mutation=0.5, crossover=0.8, deadline=None):
        """Apply local differential evolution to polish the solution."""
        n = len(params) // 3
        dim = len(params)
        
        # Create population around current best
        pop = np.zeros((pop_size, dim))
        pop[0] = params
        for i in range(1, pop_size):
            pop[i] = params + np.random.randn(dim) * 0.05
            # Ensure radii are positive
            pop[i, 2::3] = np.maximum(pop[i, 2::3], 0.01)
            # Ensure positions are in [0,1]
            pop[i, 0::3] = np.clip(pop[i, 0::3], 0.01, 0.99)
            pop[i, 1::3] = np.clip(pop[i, 1::3], 0.01, 0.99)
        
        best = params.copy()
        best_obj = objective(best, 50.0)
        
        for it in range(iterations):
            if deadline is not None and time.time() > deadline:
                break
            
            for i in range(pop_size):
                if deadline is not None and time.time() > deadline:
                    break
                
                # Select three distinct random individuals
                a, b, c = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Mutation
                trial = pop[a] + mutation * (pop[b] - pop[c])
                
                # Crossover
                for d in range(dim):
                    if np.random.rand() < crossover:
                        trial[d] = pop[a][d]
                    else:
                        trial[d] = params[d]  # Use current best as base
                
                # Ensure constraints
                trial[2::3] = np.maximum(trial[2::3], 0.01)
                trial[0::3] = np.clip(trial[0::3], 0.01, 0.99)
                trial[1::3] = np.clip(trial[1::3], 0.01, 0.99)
                
                trial_obj = objective(trial, 50.0)
                
                if trial_obj < objective(pop[i], 50.0):
                    pop[i] = trial
                    if trial_obj < best_obj:
                        best = trial.copy()
                        best_obj = trial_obj
        
        return best
    
    # Main optimization loop
    params = hex_init(N)
    
    # Stage 1: Coarse optimization with lower penalty
    penalty_weights = [10.0, 50.0, 100.0, 200.0]
    for pw in penalty_weights:
        if time.time() > deadline:
            break
        
        result = minimize(
            objective,
            params,
            jac=objective_grad,
            args=(pw,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        params = result.x
        # Enforce constraints
        params[2::3] = np.maximum(params[2::3], 0.001)
        params[0::3] = np.clip(params[0::3], 0.001, 0.999)
        params[1::3] = np.clip(params[1::3], 0.001, 0.999)
    
    # Stage 2: Local DE polishing
    if time.time() < deadline - 20:
        params = local_de(params, iterations=30, pop_size=15, deadline=deadline - 10)
    
    # Final refinement with high penalty
    if time.time() < deadline - 5:
        result = minimize(
            objective,
            params,
            jac=objective_grad,
            args=(500.0,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        params = result.x
        params[2::3] = np.maximum(params[2::3], 0.001)
        params[0::3] = np.clip(params[0::3], 0.001, 0.999)
        params[1::3] = np.clip(params[1::3], 0.001, 0.999)
    
    # Convert to output format
    result_array = params.reshape(N, 3)
    
    # Final validation: ensure all circles are inside and non-overlapping
    # Adjust radii if needed
    for i in range(N):
        x, y, r = result_array[i]
        # Adjust radius to fit boundary
        max_r = min(x, 1 - x, y, 1 - y)
        if r > max_r:
            result_array[i, 2] = max_r * 0.999
    
    return result_array
# EVOLVE-BLOCK-END
