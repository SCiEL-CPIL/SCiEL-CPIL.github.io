"""
Strategy: Binary search on radius + Feasibility-based local optimization.
Instead of maximizing the minimum distance directly (which can be noisy),
we guess a target radius R and try to find a configuration where all
pairwise distances >= 2R and wall distances >= R.
We minimize the "violation" function: max(0, 2R - d_ij) and max(0, R - w_i).
If violation is 0, R is feasible. We binary search for the max feasible R.
"""

import numpy as np
import time
import random

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _violation(cents, target_r):
    """
    Calculate the maximum violation of the constraints for a given target radius.
    Violation = max( max(0, 2*target_r - dist_ij), max(0, target_r - wall_dist_i) )
    Returns 0 if the configuration is valid for target_r.
    """
    n = len(cents)
    # Wall violations
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    wall_violations = np.maximum(0, target_r - wall_dists)
    max_wall_viol = np.max(wall_violations)
    
    # Pairwise violations
    # Only compute if n > 1
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # We only care about pairs where dist < 2*target_r
        # But for gradient, we need to know which pairs are violating.
        # Let's compute dists for all pairs.
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        pair_violations = np.maximum(0, 2.0 * target_r - dists)
        max_pair_viol = np.max(pair_violations)
    else:
        max_pair_viol = 0.0
        
    return max(max_wall_viol, max_pair_viol)

def _optimize_for_radius(cents, target_r, time_limit=5.0, rng=None):
    """
    Try to find a configuration that is feasible for target_r.
    Uses a gradient-based method to minimize the violation.
    If violation reaches 0 (or close), return the config.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    start_time = time.time()
    current_cents = cents.copy()
    current_viol = _violation(current_cents, target_r)
    
    # If already feasible, return
    if current_viol < 1e-10:
        return current_cents, 0.0, True

    # Parameters for optimization
    # We use a simple gradient descent with momentum on the violation function.
    # The violation function is not differentiable everywhere, but we can approximate the gradient.
    # Alternatively, use a force-based approach that only pushes violating pairs apart.
    
    dt = 0.01
    learning_rate = 0.05
    momentum = 0.9
    v = np.zeros_like(current_cents)
    
    # We will iterate until time limit
    while time.time() - start_time < time_limit:
        # Calculate forces based on violations
        forces = np.zeros_like(current_cents)
        
        # Wall forces: if wall_dist < target_r, push inwards
        wall_dists_x = 0.5 - np.abs(current_cents[:, 0])
        wall_dists_y = 0.5 - np.abs(current_cents[:, 1])
        
        # If x < 0, wall_dist is 0.5 + x. If x > 0, wall_dist is 0.5 - x.
        # Violation: target_r - wall_dist > 0
        # Force direction: towards center.
        # If current_cents[:, 0] < 0, push +x. If > 0, push -x.
        
        viol_x = np.maximum(0, target_r - (0.5 - current_cents[:, 0])) # Left wall
        viol_x2 = np.maximum(0, target_r - (0.5 + current_cents[:, 0])) # Right wall
        # Actually, simpler:
        # dist_to_left = current_cents[:, 0] + 0.5
        # dist_to_right = 0.5 - current_cents[:, 0]
        # min_dist = min(dist_to_left, dist_to_right)
        # viol = max(0, target_r - min_dist)
        
        dist_left = current_cents[:, 0] + 0.5
        dist_right = 0.5 - current_cents[:, 0]
        min_dist_x = np.minimum(dist_left, dist_right)
        viol_x = np.maximum(0, target_r - min_dist_x)
        
        # Gradient of viol_x wrt x:
        # if min_dist_x == dist_left (x < 0), viol = target_r - (x + 0.5) => d/dx = -1
        # if min_dist_x == dist_right (x > 0), viol = target_r - (0.5 - x) => d/dx = +1
        # So force is -grad * lr = +1 if x<0, -1 if x>0.
        # We want to move x to increase min_dist_x.
        # If x < 0, increase x. If x > 0, decrease x.
        # So force direction is -sign(x).
        
        f_wall_x = -np.sign(current_cents[:, 0]) * viol_x
        # Handle x=0: random direction? Or 0.
        f_wall_x[np.abs(current_cents[:, 0]) < 1e-6] = 0
        
        dist_bottom = current_cents[:, 1] + 0.5
        dist_top = 0.5 - current_cents[:, 1]
        min_dist_y = np.minimum(dist_bottom, dist_top)
        viol_y = np.maximum(0, target_r - min_dist_y)
        
        f_wall_y = -np.sign(current_cents[:, 1]) * viol_y
        f_wall_y[np.abs(current_cents[:, 1]) < 1e-6] = 0
        
        forces[:, 0] += f_wall_x
        forces[:, 1] += f_wall_y
        
        # Pairwise forces
        diff = current_cents[:, np.newaxis, :] - current_cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e-6)
        
        # Violation: max(0, 2*target_r - dist)
        # We want to increase dist.
        # Force on i from j: direction (i - j) / dist * viol_ij
        viol_pairs = np.maximum(0, 2.0 * target_r - dists)
        np.fill_diagonal(viol_pairs, 0)
        
        # Normalized direction
        # diff / dists
        dir_pairs = diff / (dists[:, :, np.newaxis] + 1e-9)
        
        # Force = sum_j viol_ij * dir_ij
        f_pairs = np.sum(viol_pairs[:, :, np.newaxis] * dir_pairs, axis=1)
        
        forces += f_pairs
        
        # Update velocity and position
        v = momentum * v + forces * learning_rate
        new_cents = current_cents + v * dt
        
        # Clamp to square (hard constraint to keep valid)
        new_cents = np.clip(new_cents, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_viol = _violation(new_cents, target_r)
        
        if new_viol < current_viol:
            current_cents = new_cents
            current_viol = new_viol
            if current_viol < 1e-10:
                return current_cents, 0.0, True
        else:
            # Revert if worse
            # Optionally add noise to escape local minima
            if rng.random() < 0.1:
                current_cents += rng.normal(0, 0.01, size=current_cents.shape)
                current_cents = np.clip(current_cents, -0.5 + 1e-9, 0.5 - 1e-9)
                current_viol = _violation(current_cents, target_r)
        
        # Adaptive learning rate?
        # If stuck, reduce learning rate?
        # For now, keep simple.
        
    return current_cents, current_viol, False

def _random_initialization(rng):
    """Generate a random initial configuration."""
    # Random points in the square, slightly inset
    inset = 0.05
    x = rng.uniform(-0.5 + inset, 0.5 - inset, N)
    y = rng.uniform(-0.5 + inset, 0.5 - inset, N)
    return np.column_stack((x, y))

def _hexagonal_initialization():
    """Generate a hexagonal initial configuration."""
    # This is a heuristic initialization
    best_cents = None
    best_r = 0
    
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Step 1: Get a good initial guess using hexagonal packing
    cents_hex, r_hex = _hexagonal_initialization()
    best_cents = cents_hex.copy()
    best_r = r_hex
    
    # Step 2: Binary search for the maximum feasible radius
    # Lower bound: current best radius
    # Upper bound: theoretical max (approx 0.075 for N=50? Let's use 0.1 to be safe)
    r_low = best_r
    r_high = 0.1
    
    # We will try to improve r_low by testing higher radii
    # But binary search requires a feasibility oracle.
    # Our oracle is _optimize_for_radius.
    # If it succeeds (violation ~ 0), r is feasible.
    
    # Let's do a few iterations of binary search
    num_binary_steps = 10
    
    for _ in range(num_binary_steps):
        if time.time() > deadline - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible config for r_mid
        # Use the current best config as the starting point for optimization
        cents_opt, viol, feasible = _optimize_for_radius(best_cents, r_mid, time_limit=5.0, rng=rng)
        
        if feasible:
            # r_mid is feasible. Update best if score is better
            r_actual = _score(cents_opt)
            if r_actual > best_r:
                best_r = r_actual
                best_cents = cents_opt.copy()
            r_low = r_mid
        else:
            # r_mid is not feasible.
            # Use the optimized config as the new starting point?
            # It might be closer to feasible for slightly smaller r.
            # But for binary search, we just update r_high.
            r_high = r_mid
            
            # Optional: Update best_cents with the best attempt if it's better than current best?
            # The violation config might have a higher score than the previous best?
            # Unlikely if it was infeasible for r_mid > best_r.
            # But if r_mid was close, maybe.
            r_attempt = _score(cents_opt)
            if r_attempt > best_r:
                best_r = r_attempt
                best_cents = cents_opt.copy()
                r_low = best_r # Update lower bound to the actual best found
    
    # Step 3: Local refinement around the best found radius
    # Use the best radius found and try to optimize a bit more with multiple restarts
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        # Try a few random restarts from the best config
        for i in range(3):
            if time.time() > deadline - 2:
                break
            # Perturb the best config
            perturbed = best_cents + rng.normal(0, 0.005, size=best_cents.shape)
            perturbed = np.clip(perturbed, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Try to optimize for a radius slightly higher than current best
            r_target = best_r * 1.01
            cents_opt, viol, feasible = _optimize_for_radius(perturbed, r_target, time_limit=min(remaining_time/3, 10), rng=rng)
            
            r_actual = _score(cents_opt)
            if r_actual > best_r:
                best_r = r_actual
                best_cents = cents_opt.copy()
                
    return best_cents
