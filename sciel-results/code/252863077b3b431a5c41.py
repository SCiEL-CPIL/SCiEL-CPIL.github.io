"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    # Use efficient computation
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _energy(centers, r_target):
    """
    Compute the energy (sum of squared violations) for a given target radius.
    Violations are:
    1. Pairwise distance < 2 * r_target
    2. Boundary distance < r_target
    """
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violations: max(0, 2*r_target - dist)^2
    pair_viol = np.maximum(0, 2 * r_target - dists)
    pair_energy = np.sum(pair_viol ** 2)
    
    # Boundary violations: max(0, r_target - (0.5 - |c|))^2
    # margin = 0.5 - max(|cx|, |cy|)
    margins = 0.5 - np.max(np.abs(centers), axis=1)
    bound_viol = np.maximum(0, r_target - margins)
    bound_energy = np.sum(bound_viol ** 2)
    
    return pair_energy + bound_energy

def _sa_optimize(centers, r_target, start_time, deadline):
    """
    Simulated Annealing to minimize energy for a fixed target radius r_target.
    Returns (best_centers, best_energy).
    """
    current_centers = centers.copy()
    current_energy = _energy(current_centers, r_target)
    
    best_centers = current_centers.copy()
    best_energy = current_energy
    
    # SA Parameters
    T = 0.001  # Initial temperature (energy scale)
    cooling_rate = 0.995
    min_T = 1e-6
    
    # Step size for moves
    base_step = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a move: pick a random point and displace it
        idx = np.random.randint(N)
        # Displacement magnitude scales with T to allow larger jumps early
        step_size = base_step * (1.0 + T)
        
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_centers = current_centers.copy()
        new_centers[idx] += np.array([dx, dy])
        
        # Hard clamp to square to keep points inside (prevents huge boundary violations)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_energy = _energy(new_centers, r_target)
        
        delta_E = new_energy - current_energy
        
        # Acceptance probability
        if delta_E < 0:
            # Always accept improvement
            current_centers = new_centers
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            if np.random.random() < np.exp(-delta_E / T):
                current_centers = new_centers
                current_energy = new_energy
        
        T *= cooling_rate
        
    return best_centers, best_energy

def construct():
    start_time = time.time()
    # Leave 10s safety margin for the final scoring and return
    deadline = start_time + 140
    
    # 1. Initialize with a hexagonal grid
    # Estimate spacing for 200 points
    # Area per point = 1/200 = 0.005
    # Hex spacing d: sqrt(3)/2 * d^2 = 0.005 => d^2 = 0.01 / sqrt(3) ~ 0.00577 => d ~ 0.076
    # Let's use a slightly tighter start and let SA relax it
    d_est = 0.076
    points = []
    # Generate enough points
    for row in range(20):
        y = (row - 9.5) * d_est * np.sqrt(3) / 2
        offset = 0 if row % 2 == 0 else d_est / 2
        for col in range(20):
            x = (col - 9.5) * d_est + offset
            points.append([x, y])
    points = np.array(points[:N])
    # Center
    points -= points.mean(axis=0)
    
    # 2. Binary search on radius r
    # We want to find the largest r such that there exists a configuration with energy 0 (or very small)
    # Lower bound: current best known ~ 0.0366
    # Upper bound: theoretical max for square packing is 1/(2*sqrt(200)) ~ 0.0353? No, hex is denser.
    # Max possible r if perfectly packed in area 1: sqrt(1/(200*pi)) ~ 0.0399
    # Let's search between 0.0360 and 0.0390
    
    r_low = 0.0360
    r_high = 0.0390
    
    best_overall_centers = points
    best_overall_score = _score(points)
    
    # We will do a few binary search steps
    # Each step runs SA for a fraction of the time
    # To be robust, we run SA for the *highest* r that we can "solve" (minimize energy significantly)
    # Actually, a simpler robust approach:
    # Run SA for a specific target r. If we get energy close to 0, we can try higher r.
    # If we fail to reduce energy, we lower r.
    
    # Let's do a manual binary search with 3-4 iterations to stay within time limits
    # Time budget per SA run: ~30s
    
    num_trials = 4
    trial_time = 30.0
    
    for i in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Run SA for this r_mid
        sa_deadline = time.time() + trial_time
        centers, energy = _sa_optimize(points, r_mid, start_time, sa_deadline)
        
        # Check if we found a valid packing (energy < threshold)
        # Threshold should be small, e.g., 1e-6
        threshold = 1e-4
        
        if energy < threshold:
            # Valid packing found for r_mid
            # Update best if score is better
            score = _score(centers)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_centers = centers
            # Try higher radius
            r_low = r_mid
            # Use this configuration as start for next iteration? 
            # It's a good start, but might be biased. Let's stick with the original hex grid 
            # or the best found so far. Using best found so far is usually better for SA.
            points = best_overall_centers
        else:
            # Failed to find valid packing
            # Try lower radius
            r_high = r_mid
    
    # Final polish: Run a short SA on the best found configuration with a slightly higher target 
    # to see if we can squeeze out a bit more, or just return the best valid one.
    # Actually, the binary search converges to the max r.
    # Let's do one final local optimization on best_overall_centers to clean up any tiny numerical drifts
    # using a very low temperature SA or just gradient descent on the score.
    
    # Simple local refinement: minimize energy for r = best_overall_score
    if time.time() < deadline - 5:
        r_target = best_overall_score
        final_deadline = deadline - 5
        # Use a very low temperature to just settle
        # We can reuse _sa_optimize but with a tighter T start
        # Or just return best_overall_centers
        
    return best_overall_centers

# EVOLVE-BLOCK-END
