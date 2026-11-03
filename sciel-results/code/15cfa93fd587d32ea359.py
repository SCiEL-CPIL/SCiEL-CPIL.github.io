import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    best_radius = 0.0
    best_centers = None
    
    def compute_radius(centers):
        """
        Compute the largest common radius for given centers.
        Centers are (N, 3).
        Returns r such that spheres of radius r fit.
        """
        if len(centers) < 2:
            return float(np.min(0.5 - np.abs(centers)))
        
        # Pairwise distances
        # Use broadcasting, but for N=100 this is 100x100x3, which is small
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        dist_to_walls = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dist_to_walls)
        
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def compute_violation_energy(centers, target_r):
        """
        Compute energy based on violations of target_r.
        E = sum(max(0, 2*target_r - dist_ij)^2) + sum(max(0, target_r - wall_dist_i)^2)
        Lower is better. 0 means valid.
        """
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Violations: we want dist >= 2*target_r
        # Let d_min = 2*target_r
        # violation_ij = max(0, 2*target_r - dists_ij)
        min_dist_required = 2.0 * target_r
        violations_pair = np.maximum(0.0, min_dist_required - dists)
        # Square the violations
        energy_pair = np.sum(violations_pair**2)
        
        # Walls
        wall_dists = 0.5 - np.abs(centers)
        violations_wall = np.maximum(0.0, target_r - wall_dists)
        energy_wall = np.sum(violations_wall**2)
        
        return float(energy_pair + energy_wall)

    def force_based_search(centers, steps=2000, lr_start=0.05, lr_end=0.001):
        """
        Deterministic local search using repulsive forces.
        """
        centers = centers.copy()
        centers = np.clip(centers, -0.5, 0.5)
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
            
            # Compute forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1.0)
            
            inv_dists_sq = 1.0 / (dists_sq + 1e-6)
            np.fill_diagonal(inv_dists_sq, 0.0)
            
            sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
            sum_c_j_over_d_sq = inv_dists_sq @ centers
            
            F_spheres = centers * sum_inv_dists_sq[:, np.newaxis] - sum_c_j_over_d_sq
            
            # Wall forces
            abs_c = np.abs(centers)
            d_wall = 0.5 - abs_c
            d_wall = np.maximum(d_wall, 1e-6)
            F_walls = -np.sign(centers) / (d_wall**2)
            
            F_total = F_spheres + F_walls
            
            lr = lr_start * (lr_end / lr_start) ** (step / steps)
            
            centers += lr * F_total
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    def simulated_annealing(centers, target_r, time_limit=5.0):
        """
        Stochastic local search to satisfy target_r.
        """
        centers = centers.copy()
        current_energy = compute_violation_energy(centers, target_r)
        best_energy = current_energy
        best_centers = centers.copy()
        
        T = 0.01  # Initial temperature
        T_min = 0.0001
        cooling_rate = 0.995
        
        # Estimate scale of moves
        # Typical distance between centers is ~ 2*target_r
        # Move size should be comparable
        move_scale = 0.5 * target_r + 0.01
        
        start_sa_time = time.time()
        
        while time.time() < start_sa_time + time_limit:
            # Propose move: pick random particle, random Gaussian displacement
            i = np.random.randint(N)
            displacement = np.random.normal(0, move_scale, 3)
            
            new_centers = centers.copy()
            new_centers[i] += displacement
            # Clip to cube
            new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
            
            new_energy = compute_violation_energy(new_centers, target_r)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                centers = new_centers
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_centers = centers.copy()
                    
            # Cool down
            T *= cooling_rate
            if T < T_min:
                # Reheat slightly if stuck? Or just stop cooling.
                # For now, keep T at T_min to allow random walks
                T = T_min
                
        return best_centers, best_energy

    # --- Main Logic ---
    
    # 1. Generate initial candidates
    candidates = []
    
    # Grid 5x5x4
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # Grid 4x5x5
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # Random
    np.random.seed(42)
    candidates.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # Jittered Grid
    base_grid = candidates[0].copy()
    np.random.seed(123)
    jitter = np.random.normal(0, 0.01, (N, 3))
    candidates.append(np.clip(base_grid + jitter, -0.5, 0.5))
    
    # Initial refinement with force-based search to get a good starting point
    for i, init_centers in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        
        # Quick force-based refinement
        opt_centers = force_based_search(init_centers, steps=1000)
        r = compute_radius(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
            
    # 2. Iterative improvement with Simulated Annealing
    # We try to push the radius up by setting a target_r slightly above current best
    # and using SA to find a configuration that satisfies it.
    
    current_best_centers = best_centers.copy() if best_centers is not None else candidates[0]
    current_best_radius = best_radius
    
    # If no valid center yet, initialize
    if current_best_radius == 0.0:
        current_best_centers = candidates[0]
        current_best_radius = compute_radius(current_best_centers)
        
    # Loop increasing target radius
    # Known best is ~0.11138. Let's start from current best and go up.
    # If current best is low, we might need to jump.
    
    # Safety: ensure we don't start with a target that is impossible
    # Start target slightly above current best
    target_r = current_best_radius * 1.01 + 0.001
    
    max_iterations = 20
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        # Run SA for a short duration
        # Allocate remaining time divided by remaining iterations, capped
        time_left = deadline - time.time() - 10.0
        sa_time = min(5.0, time_left / (max_iterations - it))
        
        if sa_time < 0.5:
            break
            
        sa_centers, sa_energy = simulated_annealing(current_best_centers, target_r, time_limit=sa_time)
        
        # Check if SA found a valid configuration
        # Valid means energy is very small (close to 0)
        # Or simply compute the radius of the SA result
        r_sa = compute_radius(sa_centers)
        
        # Also try force-based polish on the SA result
        polished_centers = force_based_search(sa_centers, steps=500)
        r_polished = compute_radius(polished_centers)
        
        # Take the best of SA and Polished
        if r_polished > r_sa:
            best_candidate_centers = polished_centers
            best_candidate_radius = r_polished
        else:
            best_candidate_centers = sa_centers
            best_candidate_radius = r_sa
            
        if best_candidate_radius > current_best_radius:
            # Improvement!
            current_best_radius = best_candidate_radius
            current_best_centers = best_candidate_centers.copy()
            # Increase target for next iteration
            target_r = current_best_radius * 1.005 + 0.0005
        else:
            # No improvement, maybe target was too high or we are stuck.
            # Try a slightly lower target to see if we can refine, or keep same.
            # If we are stuck, we might need to randomize the start for SA more.
            # For now, just keep the same target and hope SA explores better next time.
            # Or decrease target slightly to allow easier optimization?
            # Let's stick with the same target but maybe increase move scale?
            # The SA function uses fixed move_scale. 
            # If no improvement for a few iterations, we might break.
            pass

    # Final polish with more steps if time permits
    if time.time() < deadline - 2.0 and current_best_centers is not None:
        final_centers = force_based_search(current_best_centers, steps=1000, lr_start=0.01, lr_end=0.0001)
        r_final = compute_radius(final_centers)
        if r_final > current_best_radius:
            current_best_radius = r_final
            current_best_centers = final_centers

    if current_best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        current_best_centers = grid[:N]

    return current_best_centers
