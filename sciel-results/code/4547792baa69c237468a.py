"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Helper: Compute potential energy (penalty for overlaps and boundary violations)
    def compute_energy(pts, r_target):
        # Vectorized pairwise distances
        # pts is (N, 2)
        # diff is (N, N, 2) -> memory heavy for 300? 300*300*2*8 = 1.44 MB. Acceptable.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Pairwise penalty: if dist < 2*r_target, penalty = (2*r_target - dist)^2
        # We want to maximize min_dist, so we minimize overlap.
        # Let's use a soft penalty that drives min_dist up.
        # Actually, standard approach: minimize sum of (r_i + r_j - d_ij)^2 for d_ij < r_i + r_j
        # Here r_i = r_target.
        # To be efficient, we only care about pairs closer than 2*r_target.
        # But vectorized is fine.
        
        # Boundary penalty: if |p| > 1 - r_target, penalty = (|p| - (1 - r_target))^2
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        boundary_violation = np.maximum(0, radii - (1.0 - r_target))
        boundary_penalty = np.sum(boundary_violation ** 2)
        
        # Pairwise overlap penalty
        # We want to push points apart if they are too close.
        # Threshold: 2 * r_target
        # Penalty: (2 * r_target - dist)^2 where dist < 2 * r_target
        overlap_mask = dist < (2.0 * r_target)
        # To avoid division by zero or issues, we just square the deficit
        deficit = np.where(overlap_mask, (2.0 * r_target - dist), 0.0)
        pairwise_penalty = np.sum(deficit ** 2)
        
        return pairwise_penalty + 10.0 * boundary_penalty

    def compute_min_radius(pts):
        # Compute actual min radius for the configuration
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        max_radius = np.max(radii)
        
        r_pair = min_pair_dist / 2.0
        r_bound = 1.0 - max_radius
        
        return min(r_pair, r_bound)

    def optimize(pts, r_target, max_time, T_init, T_final, cooling_rate):
        """
        Simulated annealing / Gradient descent hybrid to improve configuration
        while trying to maintain or exceed r_target.
        Returns improved pts.
        """
        current_pts = pts.copy()
        current_energy = compute_energy(current_pts, r_target)
        best_pts = current_pts.copy()
        best_energy = current_energy
        best_r = compute_min_radius(best_pts)
        
        # If best_r is already >= r_target, we are good, but we might want to push further?
        # The task is to maximize r. 
        # Strategy: 
        # 1. Find a valid configuration for a given r.
        # 2. Binary search on r? Or just iterate with increasing r?
        # Simpler: Start with a rough r, optimize, get best_r. 
        # Then set r_target = best_r * 1.01 and optimize again?
        # Or just run SA to minimize energy for a fixed r_target, and see what min_dist we get.
        
        # Let's use a "push" strategy. 
        # We want to minimize energy where energy penalizes overlaps at radius r_target.
        # If we can reach energy ~0, then r_target is achievable.
        
        T = T_init
        lr = 0.01
        time_limit = max_time
        
        while time.time() < time_limit:
            # Compute gradient-like forces
            # Repulsion from overlaps
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            # Force = -grad(energy)
            # Energy_pair = sum( (2r - d)^2 ) for d < 2r
            # dE/dd = 2(2r - d)(-1) = 2(d - 2r)
            # Force on i from j: direction (p_i - p_j) / d * dE/dd?
            # Let's just use a simple repulsive force: F = (2r - d) * (p_i - p_j)/d for d < 2r
            # This is proportional to the gradient of the squared overlap.
            
            forces = np.zeros_like(current_pts)
            # Vectorized force calculation is tricky with masks in pure numpy without loops for 300x300?
            # Actually, we can do it.
            # d_ij is symmetric.
            # Force on i from j: k * (2r - d_ij) * (p_i - p_j) / d_ij
            # Let's compute all pairwise forces.
            
            # To save time, only compute for pairs < 2r
            mask = dist < (2.0 * r_target)
            
            # (2r - d) is scalar per pair
            strength = np.where(mask, (2.0 * r_target - dist) / dist, 0.0)
            # strength[i,j] is the scalar multiplier for vector (p_i - p_j)
            # Wait, force on i is sum over j of strength[i,j] * (p_i - p_j)
            # Note: strength[i,j] = strength[j,i]
            
            # Force_i = sum_j strength[i,j] * (p_i - p_j)
            # = p_i * sum_j strength[i,j] - sum_j strength[i,j] * p_j
            
            # sum_j strength[i,j] is row sum of strength
            row_sums = np.sum(strength, axis=1)
            
            # sum_j strength[i,j] * p_j is matrix-vector product: strength @ p
            weighted_sum = strength @ current_pts
            
            forces = current_pts * row_sums[:, np.newaxis] - weighted_sum
            
            # Boundary force
            radii = np.sqrt(np.sum(current_pts ** 2, axis=1))
            boundary_violation = np.maximum(0, radii - (1.0 - r_target))
            # Force inward if violated
            # F_bound = -k * boundary_violation * p / |p|
            # p / |p| is unit vector outward. We want inward, so -p/|p|
            # If radii is 0, skip.
            valid_rad = radii > 1e-10
            unit_rad = np.zeros_like(current_pts)
            unit_rad[valid_rad] = current_pts[valid_rad] / radii[valid_rad, np.newaxis]
            
            boundary_forces = -boundary_violation[:, np.newaxis] * unit_rad * 10.0 # Strong boundary force
            forces += boundary_forces
            
            # Update
            # Add noise for SA
            noise = np.random.normal(0, T, size=current_pts.shape)
            current_pts += lr * forces + noise
            
            # Clamp to container? 
            # If we clamp, we might break the "smooth" optimization, but it prevents invalid states.
            # The evaluator checks validity. If we return invalid, score is 0.
            # So we MUST ensure final result is valid.
            # During optimization, we can allow temporary violations if energy penalizes them,
            # but it's safer to clamp to [-(1-r), 1-r] roughly?
            # Actually, if we clamp, we remove the boundary force effect.
            # Let's clamp to 1.0 to ensure we stay in unit disk, but the penalty handles the margin.
            # Wait, if I clamp to 1.0, the circle might stick out.
            # Better to clamp to 1.0 - r_target * 0.5? No, that might be too restrictive.
            # Let's just rely on the strong boundary force. 
            # But to be safe against "non-finite" or "outside container" errors during intermediate steps?
            # The function `construct` only returns at the end. Intermediate states don't matter to the evaluator.
            # However, if `current_pts` explodes, `compute_energy` might fail or take long.
            # Let's add a safety clamp to radius 2.0.
            radii_curr = np.sqrt(np.sum(current_pts ** 2, axis=1))
            too_far = radii_curr > 2.0
            if np.any(too_far):
                current_pts[too_far] *= 2.0 / radii_curr[too_far, np.newaxis]
            
            new_energy = compute_energy(current_pts, r_target)
            
            # Acceptance criteria (Metropolis)
            if new_energy < current_energy:
                current_energy = new_energy
                # Check if this is a better valid configuration?
                # "Better" means higher min_radius.
                curr_r = compute_min_radius(current_pts)
                if curr_r > best_r:
                    best_r = curr_r
                    best_pts = current_pts.copy()
            else:
                # Accept with probability exp(-deltaE / T)
                delta_e = new_energy - current_energy
                if np.random.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                    current_pts = current_pts # already updated
            
            # Cooling
            T *= cooling_rate
            if T < T_final:
                T = T_final
            
            # Occasionally increase learning rate if stuck?
            # Keep it simple for now.
            
        return best_pts, best_r

    # 1. Initializations
    inits = []
    
    # A: Hex lattice
    r_hex = 0.05
    spacing = 2.0 * r_hex
    k = int(1.0 / spacing) + 2
    pts_hex = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r_hex:
                pts_hex.append((x, y))
    pts_hex = np.array(pts_hex)
    if len(pts_hex) >= N:
        dists = np.hypot(pts_hex[:, 0], pts_hex[:, 1])
        idx = np.argsort(dists, kind="stable")[:N]
        inits.append(pts_hex[idx])
    
    # B: Random uniform in disk
    for _ in range(3):
        r = np.sqrt(np.random.random(N)) * (1.0 - 0.05)
        theta = np.random.random(N) * 2 * np.pi
        pts_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(pts_rand)
        
    # C: Ring based
    # 20 rings of 15?
    # Let's just do a spiral
    theta = np.linspace(0, 20 * np.pi, N)
    r = np.linspace(0.1, 0.9, N)
    pts_spiral = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    inits.append(pts_spiral)
    
    # 2. Optimization Loop
    best_overall_r = 0.0
    best_overall_pts = None
    
    # Start with a conservative r_target to ensure validity
    # The known best is ~0.053.
    # Let's try to push from 0.04 up to 0.054.
    
    # We will run multiple "sessions" with increasing r_target.
    # Each session gets a time budget.
    
    # Total time budget: 145s.
    # Let's divide time among r_targets.
    # Or, simpler: 
    # Pick a random init, optimize with r_target = 0.05.
    # If it succeeds (min_r > 0.05), try r_target = 0.051 with that result.
    # This "walking" approach is robust.
    
    current_best_r = 0.0
    current_best_pts = None
    
    # Initialize with the best of the inits based on immediate score
    for init in inits:
        r_init = compute_min_radius(init)
        if r_init > current_best_r:
            current_best_r = r_init
            current_best_pts = init.copy()
            
    if current_best_pts is None:
        # Fallback
        current_best_pts = np.zeros((N, 2))
        current_best_r = 0.0

    # Iterative refinement
    # We will try to increase the radius in small steps.
    # Step size: 0.001
    # Max steps: (0.054 - current_best_r) / 0.001 ~ 20 steps
    
    # Time per step: ~ 145 / 20 = 7s?
    # That's very tight for 300 points.
    # Let's try larger steps: 0.002. ~10 steps. 14s/step.
    
    # Or, run SA with a fixed r_target that is slightly above current best, 
    # and see if we can reach low energy.
    
    # Let's do 5 restarts with different inits, each trying to climb from 0.04 to 0.054.
    
    time_per_restart = (deadline - time.time()) / 5.0
    
    for restart in range(5):
        if time.time() > deadline - 10: break
        
        # Pick an init
        if restart == 0:
            pts = current_best_pts.copy()
            r_target = current_best_r + 0.001
        else:
            pts = inits[restart % len(inits)].copy()
            r_target = 0.045 # Start low for random inits to find a valid config first
        
        # Optimize
        # Time budget for this restart
        t_start = time.time()
        t_max = time_per_restart - (time.time() - t_start) # Ensure we don't exceed
        if t_max < 2: continue
        
        # Run optimization
        # We want to find a configuration with min_radius > r_target
        # So we optimize for r_target.
        
        # If we start with a valid config for r_target, energy is 0.
        # If not, we try to reduce energy.
        
        # Let's run the optimizer
        new_pts, new_r = optimize(pts, r_target, t_max, T_init=0.01, T_final=1e-4, cooling_rate=0.99)
        
        if new_r > current_best_r:
            current_best_r = new_r
            current_best_pts = new_pts.copy()
            
            # If we improved significantly, we can try to push r_target further in next iteration?
            # The loop structure handles this by using current_best for next restart if it's the best.
            # But for restarts 1-4, we are using random inits.
            # Maybe we should just keep running with the best found so far?
            # Let's modify: if restart > 0 and we found a better r, switch to using that best for subsequent "climbing" steps.
            
    # Final Polish
    # Use the best configuration found and run a final high-precision optimization 
    # with r_target = current_best_r + small_epsilon to squeeze out a bit more.
    
    if time.time() < deadline - 5:
        r_target = current_best_r + 0.0005
        new_pts, new_r = optimize(current_best_pts, r_target, deadline - time.time() - 2, T_init=0.005, T_final=1e-5, cooling_rate=0.995)
        if new_r > current_best_r:
            current_best_pts = new_pts
            current_best_r = new_r

    # Final Validation and Return
    # Ensure the returned points are strictly inside and non-overlapping?
    # The score is calculated by the evaluator. 
    # If my `current_best_pts` has overlaps, the score will be lower than `current_best_r` (which was computed by my function).
    # My `compute_min_radius` returns the min of pair_dist/2 and 1-max_dist.
    # If there are overlaps, pair_dist < 2*r, so r_pair < r.
    # So `current_best_r` is a valid upper bound on the score? 
    # Actually, `compute_min_radius` calculates the *actual* max radius for that configuration.
    # So if I return `current_best_pts`, the evaluator will calculate the score using the same formula.
    # So the score should be exactly `current_best_r` (modulo floating point differences).
    
    return current_best_pts

# EVOLVE-BLOCK-END
