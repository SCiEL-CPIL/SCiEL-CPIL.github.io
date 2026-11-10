import numpy as np
import time
from scipy.optimize import minimize

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # 2 Large at bottom
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    # 8 Small in top half (y in [0.5, 1.0])
    # Two rows of 4
    for k in range(8):
        row = k // 4  # 0 or 1
        col = k % 4   # 0 to 3
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _penalty_objective(params, n=10, tol=1e-7, penalty_weight=100.0):
    """
    Objective function for scipy.optimize.
    Minimizes -sum(sides) + penalty for violations.
    params: (n, 3) array with [cx, cy, s]
    """
    centers = params[:, :2]
    sides = params[:, 2]
    
    # Penalty for sides <= 0
    side_penalty = np.sum(np.maximum(0, -sides) * penalty_weight)
    
    # Boundary penalties
    x_min = centers[:, 0] - sides/2
    x_max = centers[:, 0] + sides/2
    y_min = centers[:, 1] - sides/2
    y_max = centers[:, 1] + sides/2
    
    bound_penalty = (
        np.sum(np.maximum(0, -tol - x_min) * penalty_weight) +
        np.sum(np.maximum(0, x_max - (1 + tol)) * penalty_weight) +
        np.sum(np.maximum(0, -tol - y_min) * penalty_weight) +
        np.sum(np.maximum(0, y_max - (1 + tol)) * penalty_weight)
    )
    
    # Overlap penalties
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    # For each pair, if dx < sep and dy < sep, there's overlap
    # Penalty for overlap: max(0, sep - dx) + max(0, sep - dy)
    overlap_penalty = np.sum(np.maximum(0, sep - dx)[mask] + np.maximum(0, sep - dy)[mask]) * penalty_weight
    
    # Objective: minimize -sum(sides) + penalties
    obj = -np.sum(sides) + side_penalty + bound_penalty + overlap_penalty
    
    return obj

def _polish_solution(centers, sides, n=10, tol=1e-7, time_budget=30.0):
    """
    Use scipy.optimize to polish a valid solution.
    Returns improved centers, sides.
    """
    start_time = time.time()
    
    # Convert to parameter vector
    params = np.zeros((n, 3))
    params[:, :2] = centers
    params[:, 2] = sides
    
    # Bounds for scipy
    # cx, cy in [0, 1], s in [0.01, 1.0]
    bounds = [(0, 1)] * (2 * n) + [(0.01, 1.0)] * n
    
    # Flatten for scipy
    x0 = params.flatten()
    
    # Try L-BFGS-B
    best_x = x0
    best_val = _penalty_objective(params, n, tol)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(123)
    
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break
            
        if restart == 0:
            x_init = x0
        else:
            # Perturb current best
            x_init = x0 + rng.normal(0, 0.01, x0.shape)
            # Clip to bounds
            for i in range(n):
                x_init[3*i] = np.clip(x_init[3*i], 0, 1)
                x_init[3*i+1] = np.clip(x_init[3*i+1], 0, 1)
                x_init[3*i+2] = np.clip(x_init[3*i+2], 0.01, 1.0)
        
        try:
            result = minimize(
                _penalty_objective,
                x_init,
                args=(n, tol),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
        
        if time.time() - start_time > time_budget:
            break
    
    # Convert back to centers, sides
    best_params = best_x.reshape((n, 3))
    new_centers = best_params[:, :2]
    new_sides = best_params[:, 2]
    
    # Clip to valid range
    for i in range(n):
        s = np.clip(new_sides[i], 0.01, 1.0)
        new_sides[i] = s
        new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
        new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
    
    # Check if the polished solution is valid
    if _is_valid_vectorized(new_centers, new_sides, tol):
        return new_centers, new_sides
    
    # If not valid, return original
    return centers, sides

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 625 # Safety margin
    
    # Best solution found so far
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    # SA Parameters
    T0 = 0.1
    T_min = 1e-6
    cooling_rate = 0.999
    max_iters_per_run = 50000
    stall_threshold = 5000 # No improvement for this many iters -> restart
    
    rng = np.random.default_rng(42)
    
    # Track if we've done polish
    polish_done = False
    
    while time.time() < deadline:
        # Initialize current solution
        # 50% chance to start from best, 50% from perturbed best
        if rng.random() < 0.5 and best_val > 3.0:
            centers = best_centers + rng.uniform(-0.02, 0.02, (n, 2))
            sides = best_sides + rng.uniform(-0.01, 0.01, n)
        else:
            centers, sides = _initial_packing_grid()
            centers = centers + rng.uniform(-0.05, 0.05, (n, 2))
            sides = sides + rng.uniform(-0.02, 0.02, n)
            
        # Ensure basic validity of start (clip to bounds)
        for i in range(n):
            s = max(0.01, min(1.0, sides[i]))
            sides[i] = s
            centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
            centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
            
        if not _is_valid_vectorized(centers, sides, tol):
            # If start is invalid, try to fix it or just start with grid
            centers, sides = _initial_packing_grid()
            
        current_val = np.sum(sides)
        T = T0
        last_improvement_iter = 0
        iter_count = 0
        
        while iter_count < max_iters_per_run and time.time() < deadline:
            iter_count += 1
            if iter_count % 1000 == 0 and time.time() > deadline:
                break
                
            # Generate neighbor
            new_centers = centers.copy()
            new_sides = sides.copy()
            
            idx = rng.integers(0, n)
            
            # Adaptive noise: larger when T is high
            noise_scale = 0.05 * (1 + T)
            
            new_centers[idx, 0] += rng.uniform(-noise_scale, noise_scale)
            new_centers[idx, 1] += rng.uniform(-noise_scale, noise_scale)
            new_sides[idx] += rng.uniform(-0.05, 0.05)
            
            # Enforce bounds
            s = new_sides[idx]
            if s <= 0.01:
                new_sides[idx] = 0.01
            elif s > 1.0:
                new_sides[idx] = 1.0
            else:
                new_sides[idx] = s
            
            s = new_sides[idx]
            new_centers[idx, 0] = np.clip(new_centers[idx, 0], s/2, 1 - s/2)
            new_centers[idx, 1] = np.clip(new_centers[idx, 1], s/2, 1 - s/2)
            
            # Check validity
            if _is_valid_vectorized(new_centers, new_sides, tol):
                new_val = np.sum(new_sides)
                delta = new_val - current_val
                
                # Acceptance
                if delta > 0 or rng.random() < np.exp(delta / T):
                    centers = new_centers
                    sides = new_sides
                    current_val = new_val
                    
                    if current_val > best_val:
                        best_val = current_val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
                        last_improvement_iter = iter_count
                        
            T *= cooling_rate
            
            # Stall detection
            if iter_count - last_improvement_iter > stall_threshold:
                break
        
        # After SA run, try to polish if we have time
        if time.time() < deadline - 60:  # Reserve 60s for polish
            polish_time_budget = min(30.0, deadline - time.time() - 10)
            if polish_time_budget > 5:
                polished_centers, polished_sides = _polish_solution(
                    best_centers, best_sides, n, tol, polish_time_budget
                )
                polished_val = np.sum(polished_sides)
                if polished_val > best_val:
                    best_val = polished_val
                    best_centers = polished_centers
                    best_sides = polished_sides
                
                # Mark polish done, don't repeat in same loop iteration
                polish_done = True
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
