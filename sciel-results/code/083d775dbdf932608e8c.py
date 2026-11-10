import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    # Safety margin: kill at 145s to allow for return overhead
    time_limit = 145.0
    
    def get_min_radius(centers):
        """Compute the largest radius r such that spheres at 'centers' are disjoint and inside the cube."""
        if centers.shape[0] < 2:
            return 0.5 - np.max(np.abs(centers))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances: distance from center to nearest wall
        # Walls are at +/- 0.5. Distance is 0.5 - |coord|
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize(centers, max_steps=10000, initial_step=0.005, repulsion_strength=1.0):
        """
        Force-directed relaxation.
        Returns (best_centers, best_radius).
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = initial_step
        no_improve_count = 0
        max_no_improve = 1000 # Reset logic
        
        for i in range(max_steps):
            # Time check
            if time.time() - start_time > time_limit - 2.0:
                break
            
            # Compute forces
            # 1. Sphere-Sphere Repulsion
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, np.inf)
            
            # Force magnitude: k / d^2. Direction: normalized diff.
            # F_ij = k * (c_i - c_j) / |c_i - c_j|^3
            # We can compute this vectorized.
            # Avoid division by zero
            dist_sq_safe = np.where(dist_sq == 0, 1e-10, dist_sq)
            # F = k * diff / dist^3 = k * diff / (dist_sq * sqrt(dist_sq))
            # Let's cap the force to prevent instability
            inv_dist_cubed = 1.0 / (dist_sq_safe * np.sqrt(dist_sq_safe))
            inv_dist_cubed = np.where(dist_sq == 0, 0.0, inv_dist_cubed)
            inv_dist_cubed = np.minimum(inv_dist_cubed, 100.0) # Cap
            
            forces = repulsion_strength * (diff * inv_dist_cubed[:, :, np.newaxis])
            forces = np.sum(forces, axis=1)
            
            # 2. Wall Repulsion
            # Push away from walls if close.
            # Walls at -0.5 and 0.5.
            # Force is linear or quadratic? Let's use quadratic repulsion from boundary.
            # d_pos = 0.5 - c_k. If d_pos < threshold, push in -k direction.
            # F_wall = k_w * (1/d^2) * sign
            # Simpler: F_wall = k_w * (0.5 - |c_k|) is wrong direction.
            # If c_k > 0, wall is at 0.5. dist = 0.5 - c_k. Force = -1/dist^2 (towards center).
            # If c_k < 0, wall is at -0.5. dist = 0.5 + c_k. Force = +1/dist^2.
            
            wall_strength = 2.0 # Stronger wall repulsion to keep them inside
            
            for k in range(3):
                c_k = centers[:, k]
                # Positive wall
                d_pos = 0.5 - c_k
                mask_pos = d_pos < 0.1
                if np.any(mask_pos):
                    f_pos = wall_strength / (d_pos[mask_pos]**2 + 1e-6)
                    f_pos = np.minimum(f_pos, 100.0)
                    forces[mask_pos, k] -= f_pos # Push towards negative k (inward)
                
                # Negative wall
                d_neg = 0.5 + c_k
                mask_neg = d_neg < 0.1
                if np.any(mask_neg):
                    f_neg = wall_strength / (d_neg[mask_neg]**2 + 1e-6)
                    f_neg = np.minimum(f_neg, 100.0)
                    forces[mask_neg, k] += f_neg # Push towards positive k (inward)
            
            # Normalize forces to prevent huge steps
            force_mags = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_mags) if np.max(force_mags) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            # Update
            centers += step * forces
            
            # Clamp to box (strictly inside to avoid numerical issues with dist=0)
            centers = np.clip(centers, -0.4999, 0.4999)
            
            # Evaluate
            new_r = get_min_radius(centers)
            
            # Adaptive Step
            if new_r > current_r:
                step = min(step * 1.05, 0.02)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.95, 0.001)
                no_improve_count += 1
            
            current_r = new_r
            
            # Stagnation handling
            if no_improve_count > max_no_improve:
                # Perturb random subset
                n_perturb = max(1, N // 10)
                idx = np.random.choice(N, n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.01
                centers = np.clip(centers, -0.4999, 0.4999)
                no_improve_count = 0
                step = 0.005
        
        return best_local_centers, best_local_r

    best_r = 0.0
    best_centers = None
    
    # Strategy: Multiple starts, coarse to fine.
    
    # 1. Grid Initialization
    # 5x5x5 grid has 125 points. We take 100.
    # A 4x5x5 grid is exactly 100.
    # Spacing: x=0.25, y=0.2, z=0.2.
    # Let's use a jittered 4x5x5 grid.
    xs = (np.arange(4) + 0.5) / 4.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(5) + 0.5) / 5.0 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    np.random.seed(1)
    init_grid = grid + np.random.uniform(-0.01, 0.01, (N, 3))
    init_grid = np.clip(init_grid, -0.49, 0.49)
    
    # 2. FCC-like Initialization
    # FCC lattice points
    s = 0.25
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k]) * s
                pts.append(base)
                pts.append(base + np.array([s/2, s/2, 0]))
                pts.append(base + np.array([s/2, 0, s/2]))
                pts.append(base + np.array([0, s/2, s/2]))
    fcc_grid = np.array(pts)
    fcc_grid -= fcc_grid.mean(axis=0)
    extent = np.max(np.abs(fcc_grid))
    if extent > 0.5:
        fcc_grid *= (0.5 / extent)
    # Sort by distance to center to pick the central 100
    dists_from_origin = np.linalg.norm(fcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_fcc = fcc_grid[order[:N]]
    init_fcc += np.random.uniform(-0.01, 0.01, (N, 3))
    init_fcc = np.clip(init_fcc, -0.49, 0.49)

    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))

    initializations = [init_grid, init_fcc, init_random]
    
    # Phase 1: Coarse Search
    # Allocate time roughly equally, but keep time for polish.
    # 145s total. 3 starts. ~40s each? Plus polish.
    # Let's run each for 3000 steps or until time runs out (with checks).
    
    for init in initializations:
        if time.time() - start_time > time_limit - 20:
            break
            
        c_opt, r_opt = optimize(init, max_steps=3000, initial_step=0.005)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt
            
        # Check time
        if time.time() - start_time > time_limit - 15:
            break

    # Phase 2: Fine Polish on the best result
    if best_centers is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 5:
            # Use smaller steps for precision
            c_polished, r_polished = optimize(best_centers, max_steps=5000, initial_step=0.002)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = c_polished

    # Fallback if something went wrong
    if best_centers is None:
        best_centers = init_grid
        
    # Final safety check: ensure valid shape and finite values
    if best_centers.shape != (N, 3) or not np.all(np.isfinite(best_centers)):
        best_centers = init_grid

    return best_centers
