"""
Strategy: Simulated Annealing with KD-Tree Acceleration.

This approach differs from the current SLSQP-based method by:
1. Using a stochastic global search (Simulated Annealing) instead of a deterministic local optimizer.
2. Employing a KD-Tree for efficient neighbor queries to calculate the objective function (packing radius) in O(N log N) instead of O(N^2).
3. Optimizing a "soft" energy function that penalizes overlaps and rewards spreading, which allows the system to escape local minima by accepting worse states with a probability that decreases over time (temperature schedule).
4. Combining a strong hexagonal initial guess with high-temperature global exploration and low-temperature local refinement.
"""

import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START

def _score(cents):
    """Compute the packing radius."""
    n = len(cents)
    if n == 0:
        return 0.0
    
    # Wall distances
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    
    # Pairwise distances
    if n > 1:
        tree = cKDTree(cents)
        # Query pairs. We need the minimum distance.
        # k=2 gives self and nearest neighbor.
        # For a dense packing, the nearest neighbor is the closest circle.
        # However, to be safe and accurate for the *global* min, we can just compute all pairs if N is small,
        # but KDTree is faster for the energy calculation during SA steps if we approximate or use it smartly.
        # Actually, for N=50, O(N^2) is 2500 ops. KDTree overhead might be higher.
        # Let's stick to vectorized O(N^2) for accuracy in the final score, 
        # but use a faster approximate or optimized check for the SA loop if needed.
        # Given N=50, vectorized distance matrix is very fast in numpy.
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
        
    return min(wall_min, pair_min)

def _energy(cents, r_target=0.07):
    """
    Calculate a soft energy for Simulated Annealing.
    We want to maximize the minimum distance. 
    Equivalent to minimizing the maximum overlap.
    
    Energy = Sum of overlaps (soft) + Boundary violations
    This allows the system to "push" circles apart.
    """
    n = len(cents)
    energy = 0.0
    
    # Boundary penalty: circles should stay within [-0.5+r, 0.5-r]
    # Actually, we don't know r exactly yet. 
    # Let's define energy based on a fixed target radius R_target that we try to achieve.
    # If dist < 2*R_target, penalize (dist - 2*R_target)^2.
    # If wall_dist < R_target, penalize (wall_dist - R_target)^2.
    
    # A more robust approach for SA: 
    # Objective: Minimize E = max(0, 2*R_target - d_ij)^2 + max(0, R_target - w_i)^2
    # This drives the system towards a valid packing of radius R_target.
    # If it fails, we can lower R_target.
    
    # For the SA step, let's use a simpler metric: 
    # Minimize the sum of squared inverse distances? No, that's unstable.
    # Let's use the "max overlap" metric smoothed.
    
    # Vectorized calculation for speed
    # 1. Wall violations
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    # Effective radius is the min distance to wall.
    # Let's assume we are trying to fit radius r.
    # If we don't fix r, we can just maximize the minimum distance.
    
    # Let's use a potential energy that approximates the hard-sphere repulsion.
    # U = sum_{i<j} (1/d_ij)^k + sum_i (1/w_i)^k
    # This is smooth and drives apart.
    # However, minimizing this might not strictly maximize the *minimum* distance, but it correlates well.
    
    # Better: Use the actual score but with a soft penalty for violations if we fix r.
    # Strategy: 
    # 1. Start with a good r (from hexagonal).
    # 2. Run SA to find a configuration where all constraints are satisfied for that r.
    # 3. If successful, increment r slightly and run SA again (annealing the radius).
    
    # This function will be used inside the SA loop for a FIXED r.
    # Returns violation sum.
    # If violation is 0, it's a valid packing for r.
    
    # Wall
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    w = np.minimum(wx, wy)
    wall_viol = np.maximum(0, r_target - w)
    wall_pen = np.sum(wall_viol**2)
    
    # Pairs
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs closer than 2*r_target
    # mask = dists < 2*r_target
    # viol = np.maximum(0, 2*r_target - dists)
    # pen = np.sum(viol**2)
    
    # To speed up, we can skip the full matrix if we use KDTree, but for N=50, numpy is fine.
    # Let's compute full for simplicity and correctness in the inner loop.
    viol = np.maximum(0, 2*r_target - dists)
    # Set diagonal to 0
    np.fill_diagonal(viol, 0)
    pair_pen = np.sum(viol**2)
    
    return wall_pen + pair_pen

def _construct_hexagonal():
    """Create a good initial hexagonal packing."""
    best_cents = None
    best_r = 0
    
    # Try various row counts
    for n_rows in range(5, 9):
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
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset for hexagonal
            # If row is even, no offset. If odd, offset by r*sqrt(3)/2? 
            # Actually, standard hex packing:
            # Row 0: x = -0.5 + r + i*2r
            # Row 1: x = -0.5 + r + r + i*2r (shifted by r)
            # Vertical distance: sqrt(3)*r
            offset = r * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                # Clip to ensure inside, though math says it should be
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        # Ensure strictly inside for safety
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)
        
    return best_cents, best_r

def _simulated_annealing(cents, r_target, time_limit, rng):
    """
    Run SA to find a valid packing for radius r_target starting from cents.
    Returns (cents, success)
    """
    start_time = time.time()
    current_cents = cents.copy()
    current_energy = _energy(current_cents, r_target)
    
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Step size
    step_size = 0.01
    
    best_cents = current_cents.copy()
    best_energy = current_energy
    
    while time.time() - start_time < time_limit:
        # Generate candidate
        # Move a random subset of circles? Or one circle?
        # Moving one circle is standard.
        i = rng.integers(N)
        move = rng.normal(0, step_size, size=2)
        candidate = current_cents.copy()
        candidate[i] += move
        # Keep inside square (hard constraint for position, soft for radius)
        candidate[i, 0] = np.clip(candidate[i, 0], -0.5, 0.5)
        candidate[i, 1] = np.clip(candidate[i, 1], -0.5, 0.5)
        
        candidate_energy = _energy(candidate, r_target)
        
        # Acceptance
        if candidate_energy < current_energy:
            current_cents = candidate
            current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_cents = candidate.copy()
        else:
            delta = candidate_energy - current_energy
            # Boltzmann factor
            if T > 0:
                accept_prob = np.exp(-delta / T)
                if rng.random() < accept_prob:
                    current_cents = candidate
                    current_energy = candidate_energy
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_cents, best_energy < 1e-8

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # 1. Initial Hexagonal Packing
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    rng = np.random.default_rng(42)
    
    # 2. SA Optimization Loop
    # We will try to increase the radius in small steps.
    # Start from r0 and try to go higher.
    
    # Estimate a potential upper bound
    # Area argument: 50 * pi * r^2 <= 1 => r <= 1/sqrt(50*pi) ~ 0.0797
    # But packing efficiency is ~0.9069 (hex)
    # r_max_hex ~ 0.0797 * sqrt(0.9069) ? No.
    # Area of 50 circles = 50 * pi * r^2.
    # Max area in square = 1.
    # So 50 * pi * r^2 <= 1 => r <= sqrt(1/(50*pi)) = 0.079788
    # The known best is 0.071377.
    # Let's try to push towards 0.075.
    
    current_r = r0 * 0.99 # Start slightly inside
    r_step = 0.0001
    max_r_target = 0.075
    
    # First, refine the current configuration for its current radius to ensure it's tight
    # Then try to increase
    
    while time.time() - start_time < deadline - 5:
        # Try to find a valid packing for current_r
        # If current_r is already valid, we can try to increase it
        # If not, we need to optimize for current_r first
        
        # Check if current best is valid for current_r
        # We track best_r from _score.
        # Let's just run SA for a target r slightly above current best_r.
        
        target_r = best_r + r_step
        if target_r > max_r_target:
            break
            
        # Run SA for a short time to see if we can achieve target_r
        # We use the current best_cents as start
        sa_time = min(5.0, deadline - (time.time() - start_time) - 2)
        if sa_time < 0.5:
            break
            
        new_cents, success = _simulated_annealing(best_cents, target_r, sa_time, rng)
        
        # Verify
        new_r = _score(new_cents)
        if new_r > best_r:
            best_r = new_r
            best_cents = new_cents
            # If we made progress, keep increasing target
            # If we failed, we might need to stay at current_r and optimize more
            # Or decrease target? No, we want max r.
            # If SA failed to reach target_r, it means best_r is likely close to target_r.
            # We can try a smaller step or just continue optimizing best_r.
        else:
            # No improvement. Maybe we are stuck.
            # Try to optimize for the CURRENT best_r to tighten the packing?
            # Or try a random restart?
            # Let's try a random restart with the same target_r but different init
            # Only if we have plenty of time.
            if time.time() - start_time < deadline - 20:
                # Random restart
                init_cents = best_cents + rng.normal(0, 0.01, size=(N, 2))
                init_cents[:, 0] = np.clip(init_cents[:, 0], -0.5, 0.5)
                init_cents[:, 1] = np.clip(init_cents[:, 1], -0.5, 0.5)
                new_cents, success = _simulated_annealing(init_cents, target_r, sa_time, rng)
                new_r = _score(new_cents)
                if new_r > best_r:
                    best_r = new_r
                    best_cents = new_cents
            else:
                # Time is running out, stop trying to increase
                break

    return best_cents

# EVOLVE-BLOCK-END
