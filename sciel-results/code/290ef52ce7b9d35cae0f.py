"""
Strategy: Simulated Annealing with Repulsive Forces
Instead of smooth surrogate optimization, we use a stochastic global search method.
We model the problem as an energy minimization where energy = -min_clearance.
We perform random moves (perturbing one circle's center) and accept them based on
the Metropolis criterion. This allows escaping local minima that trap gradient-based
methods like L-BFGS-B, especially in the rugged landscape of the min-function.
"""

import numpy as np
import time
import random

N = 50
DEADLINE = 145.0  # Safety margin from 150s

def _radius_score(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^2."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distance
    dist_to_boundary = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2
    
    return min(min_boundary, min_pairwise)

def _compute_clearances(centers):
    """
    Compute all clearances: boundary distances and half pairwise distances.
    Returns a tuple (min_clearance, all_clearances_array) for efficiency in SA.
    """
    # Boundary clearances
    b_x_pos = centers[:, 0] + 0.5
    b_x_neg = 0.5 - centers[:, 0]
    b_y_pos = centers[:, 1] + 0.5
    b_y_neg = 0.5 - centers[:, 1]
    
    boundary_clearances = np.concatenate([b_x_pos, b_x_neg, b_y_pos, b_y_neg])
    
    # Pairwise clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    pair_clearances = dists[iu] / 2.0
    
    all_clearances = np.concatenate([boundary_clearances, pair_clearances])
    min_clearance = np.min(all_clearances)
    
    return min_clearance, all_clearances

def _simulated_annealing(initial_centers, time_budget=10.0, start_temp=0.01, end_temp=1e-6):
    """
    Run Simulated Annealing to optimize the circle packing.
    """
    start_time = time.time()
    centers = initial_centers.copy()
    
    # Initial evaluation
    current_min, _ = _compute_clearances(centers)
    best_min = current_min
    best_centers = centers.copy()
    
    # SA parameters
    # We'll adapt the step size based on current temperature
    base_step = 0.05
    
    # Cooling schedule: exponential
    # We need to estimate number of iterations possible in time_budget
    # Let's just run until time runs out, adjusting temp based on elapsed time fraction
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
            
        # Progress fraction (0 to 1)
        progress = min(1.0, elapsed / time_budget)
        
        # Temperature: exponential cooling
        temp = start_temp * np.exp(-5.0 * progress) + end_temp
        
        # Step size: decrease as we cool
        step_size = base_step * (1.0 - 0.5 * progress)
        
        # Choose a random circle to move
        idx = random.randint(0, N - 1)
        
        # Generate a random move
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        
        # New position
        new_x = centers[idx, 0] + dx
        new_y = centers[idx, 1] + dy
        
        # Keep within bounds (hard constraint)
        new_x = np.clip(new_x, -0.5, 0.5)
        new_y = np.clip(new_y, -0.5, 0.5)
        
        # Skip if no actual move
        if new_x == centers[idx, 0] and new_y == centers[idx, 1]:
            continue
            
        # Apply move temporarily
        old_x, old_y = centers[idx, 0], centers[idx, 1]
        centers[idx, 0] = new_x
        centers[idx, 1] = new_y
        
        # Evaluate new state
        new_min, _ = _compute_clearances(centers)
        
        # Acceptance criterion
        if new_min >= current_min:
            # Always accept improvement
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_centers = centers.copy()
        else:
            # Accept worse state with probability exp(-delta / temp)
            delta = current_min - new_min  # Positive value
            if delta < 1e-12:
                accept = True
            else:
                # Avoid overflow in exp
                if delta / temp > 50:
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / temp)
            
            if not accept:
                # Reject move
                centers[idx, 0] = old_x
                centers[idx, 1] = old_y
            else:
                current_min = new_min
        
        # Periodic check for time
        if time.time() - start_time > time_budget:
            break
    
    return best_centers, best_min

def _construct():
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    random.seed(42)  # For reproducibility in SA
    
    best_centers = None
    best_radius = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Grid
    m, n = 7, 8
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 2. Hex
    points = []
    row_height = 0.09
    col_width = 0.1
    offset = col_width / 2
    for row in range(8):
        y = -0.5 + 0.05 + row * row_height
        for col in range(7):
            if len(points) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 3. Random inits
    for _ in range(10):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
    
    # Run SA on each init
    # Allocate time: 10 inits, ~12s each = 120s, leaving 25s margin
    time_per_init = min(12.0, (DEADLINE - time.time()) / max(1, len(inits)))
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 5.0:
            break
            
        remaining = DEADLINE - time.time()
        if remaining < 2.0:
            break
            
        # Adjust time per init if we're running late
        t_budget = min(time_per_init, remaining - 2.0)
        
        optimized, r = _simulated_annealing(init, time_budget=t_budget)
        
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
            
    # Final polish: Run SA on the best result found with higher precision
    if best_centers is not None and time.time() < DEADLINE - 3.0:
        remaining = DEADLINE - time.time() - 1.0
        if remaining > 1.0:
            # Run a shorter, more focused SA on the best result
            polished, r = _simulated_annealing(best_centers, time_budget=min(5.0, remaining), 
                                              start_temp=0.005, end_temp=1e-7)
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()
    
    if best_centers is None:
        # Fallback: Grid
        m, n = 7, 8
        points = []
        for i in range(n):
            for j in range(m):
                if len(points) >= N:
                    break
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / n - 0.5
                points.append([x, y])
        best_centers = np.array(points[:N])
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

def construct():
    return _construct()
