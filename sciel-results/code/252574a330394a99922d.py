"""
Strategy: Differential Evolution (DE) for Circle Packing
- Objective: Maximize sum(r_i)
- Constraints: 
  1. Circles inside rectangle [0, w] x [0, h]
  2. Pairwise distance d_ij >= r_i + r_j
- Method:
  - Vectorized penalty function to handle constraints.
  - Differential Evolution to search the 63-dimensional space (21 circles * 3 params).
  - Multiple aspect ratios explored to find the optimal rectangle shape.
"""

import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 21

# EVOLVE-BLOCK-START

def _validate_and_score(x, w, h):
    """
    Computes the score (sum of radii) if valid, or -inf if invalid.
    x: (N, 3) array of [cx, cy, r]
    """
    # 1. Check bounds
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    
    # Radii must be positive
    if np.any(r <= 0):
        return -np.inf
        
    # Inside rectangle
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return -np.inf
    if np.any(cx + r > w + 1e-9) or np.any(cy + r > h + 1e-9):
        return -np.inf
        
    # 2. Check pairwise overlaps
    # Calculate all pairwise distances
    # dist[i, j] = distance between circle i and j
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dist = r[:, None] + r[None, :]
    
    # We need dist[i, j] >= min_dist[i, j] for all i != j
    # Ignore diagonal (dist=0, min_dist=2r_i > 0)
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(min_dist, 0)
    
    # Check if any violation exists
    if np.any(dist < min_dist - 1e-9):
        return -np.inf
        
    return np.sum(r)

def _penalty_objective(x_flat, w, h):
    """
    Objective function for differential_evolution (minimization).
    Returns a large penalty if constraints are violated, 
    otherwise returns -sum(radii) (so minimizing this maximizes radii).
    """
    x = x_flat.reshape((N, 3))
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    
    penalty = 0.0
    
    # 1. Boundary violations
    # If cx - r < 0, penalty += (0 - (cx - r))^2
    left_viol = np.maximum(0, r - cx)
    right_viol = np.maximum(0, cx + r - w)
    bottom_viol = np.maximum(0, r - cy)
    top_viol = np.maximum(0, cy + r - h)
    
    boundary_pen = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                   np.sum(bottom_viol**2) + np.sum(top_viol**2)
    penalty += boundary_pen * 100.0  # Weight boundary heavily
    
    # 2. Pairwise overlap violations
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    min_dist = r[:, None] + r[None, :]
    np.fill_diagonal(min_dist, 0)
    
    overlap = np.maximum(0, min_dist - dist)
    overlap_pen = np.sum(overlap**2)
    penalty += overlap_pen * 100.0
    
    # 3. Objective: Maximize sum(r) -> Minimize -sum(r)
    obj = -np.sum(r)
    
    return penalty + obj

def _run_de_for_aspect(w, h, deadline, seed):
    """
    Runs Differential Evolution for a specific aspect ratio (w, h).
    """
    # Define bounds for each variable
    # x_i in [0, w], y_i in [0, h], r_i in [0, min(w, h)/2]
    # To be safe, set r max slightly smaller to allow movement
    r_max = min(w, h) * 0.45
    
    bounds = []
    for i in range(N):
        bounds.append((0, w))   # cx
        bounds.append((0, h))   # cy
        bounds.append((0.01, r_max)) # r (avoid 0)
        
    # Flatten bounds for DE
    bounds_de = [(lb, ub) for lb, ub in bounds]
    
    # Strategy: Use a smaller population and maxiter to fit in time, 
    # but rely on multiple restarts/aspects.
    # However, DE is stochastic. We can run it for a few seconds per aspect.
    
    # We need to be careful with time. 
    # Let's run DE for a fixed number of iterations or until time runs out.
    # Since we don't know how long one iteration takes, we'll use a wrapper 
    # that stops early if needed, or just set conservative limits.
    
    # For N=21, 63 variables.
    # popsize=15, maxiter=50 might be reasonable for a quick scan.
    
    # We will run multiple DE instances with different seeds for this aspect if time permits.
    best_val = -np.inf
    best_x = None
    
    start_t = time.time()
    time_budget_per_de = 5.0 # seconds per DE run
    
    # We will try 2-3 DE runs for this aspect
    num_de_runs = 3
    
    for run_idx in range(num_de_runs):
        if time.time() > deadline - 10:
            break
            
        run_seed = seed + run_idx * 100
        
        # Differential Evolution
        # Use 'best1bin' or 'rand1bin'
        # tol=1e-4, maxiter=100, popsize=20
        
        try:
            res = differential_evolution(
                _penalty_objective, 
                bounds_de, 
                args=(w, h),
                seed=run_seed,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                polish=False, # Polish can be slow, we'll do our own local refine if needed
                mutation=(0.5, 1.0),
                recombination=0.9,
                # Update callback to check time? 
                # scipy DE doesn't have a simple time-based callback that stops cleanly 
                # without custom logic, so we rely on maxiter being small enough.
                # 100 iters * 20 pop * 63 vars is fast.
            )
            
            x_opt = res.x.reshape((N, 3))
            val = _validate_and_score(x_opt, w, h)
            
            if val > best_val:
                best_val = val
                best_x = x_opt
                
        except Exception as e:
            pass
            
    return best_x, best_val

def _local_refine(x, w, h, deadline):
    """
    Simple local refinement using gradient descent on the penalty function
    to squeeze out more sum(r) or fix small violations.
    """
    # Convert to flat
    x_flat = x.flatten()
    
    # We want to maximize sum(r) subject to penalty=0.
    # Let's just do a few steps of projected gradient or similar?
    # Actually, since DE with penalty often lands on the boundary of feasibility,
    # we can try to increase radii slightly and push positions.
    
    # This is a heuristic "grow and push" similar to the original but 
    # applied to the DE result which is likely already near-optimal.
    
    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        # Increase radii slightly
        x[:, 2] += 0.001
        
        # Resolve overlaps by pushing apart
        cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
        
        # Compute forces
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1e-6)
        
        min_dist = r[:, None] + r[None, :]
        overlap = np.maximum(0, min_dist - dist)
        
        # Force direction
        force_x = np.sum(overlap * (dx / dist), axis=1)
        force_y = np.sum(overlap * (dy / dist), axis=1)
        
        # Boundary forces
        left_viol = np.maximum(0, r - cx)
        right_viol = np.maximum(0, cx + r - w)
        bottom_viol = np.maximum(0, r - cy)
        top_viol = np.maximum(0, cy + r - h)
        
        force_x += left_viol - right_viol
        force_y += bottom_viol - top_viol
        
        # Update
        x[:, 0] += 0.1 * force_x
        x[:, 1] += 0.1 * force_y
        
        # Clip to bounds (positions)
        x[:, 0] = np.clip(x[:, 0], r, w - r)
        x[:, 1] = np.clip(x[:, 1], r, h - r)
        
    # Check validity
    val = _validate_and_score(x, w, h)
    if val == -np.inf:
        # If we broke it, revert to previous best? 
        # We don't have previous best here easily, so return None to signal failure
        return None, -np.inf
        
    return x, val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_solution = None
    best_value = -np.inf
    
    # Explore different aspect ratios
    # w + h = 2
    # aspect = w/h
    # w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    
    aspects = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.8, 0.7, 1.1, 1.4, 0.5]
    random.shuffle(aspects)
    
    for aspect in aspects:
        if time.time() > deadline - 20:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        if w <= 0 or h <= 0:
            continue
            
        # Run DE
        sol, val = _run_de_for_aspect(w, h, deadline, seed=int(aspect*100))
        
        if sol is not None and val > best_value:
            best_value = val
            best_solution = sol
            
            # Try local refinement if we have time
            if time.time() < deadline - 15:
                refined_sol, refined_val = _local_refine(sol.copy(), w, h, deadline)
                if refined_sol is not None and refined_val > best_value:
                    best_value = refined_val
                    best_solution = refined_sol
                    
    # Fallback if DE failed completely (unlikely, but possible)
    if best_solution is None or best_value <= 0:
        # Simple grid fallback
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                cx = (j + 0.5) * w / side
                cy = (i + 0.5) * h / side
                r = 0.4 / side
                cells.append([cx, cy, r])
                idx += 1
        best_solution = np.array(cells[:N])
        best_value = _validate_and_score(best_solution, w, h)
        if best_value == -np.inf:
             # Make it valid by shrinking
             best_solution[:, 2] *= 0.5
             best_value = _validate_and_score(best_solution, w, h)

    return best_solution

# EVOLVE-BLOCK-END
