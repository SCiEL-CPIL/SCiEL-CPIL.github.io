import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety (evaluator kills at 150s)
    
    best_r = 0.0
    best_centers = None
    
    def get_min_radius(centers):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize(centers, max_steps=50000, step_size=0.005, repulsion_strength=1.0):
        """Force-directed relaxation with soft repulsion from all neighbors."""
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        # Adaptive step size
        step = step_size
        no_improve_count = 0
        max_no_improve = 2000
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Compute repulsion forces from all neighbors
            inv_dist_sq = 1.0 / (dist**2 + 1e-10)
            inv_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, inv_dist_sq)
            inv_dist_sq = np.minimum(inv_dist_sq, 100.0)
            
            # Vectorized force computation
            row_sums = np.sum(inv_dist_sq, axis=1)
            weighted_centers = inv_dist_sq @ centers
            forces = centers * row_sums[:, np.newaxis] - weighted_centers
            forces *= repulsion_strength
            
            # Wall repulsion: push points away from walls
            for k in range(3):
                dist_to_pos_wall = 0.5 - centers[:, k]
                dist_to_neg_wall = 0.5 + centers[:, k]
                
                wall_force = np.zeros(N)
                
                close_to_pos = dist_to_pos_wall < 0.1
                if np.any(close_to_pos):
                    wf = repulsion_strength / (dist_to_pos_wall[close_to_pos]**2 + 1e-10)
                    wf = np.minimum(wf, 100.0)
                    wall_force[close_to_pos] += wf
                
                close_to_neg = dist_to_neg_wall < 0.1
                if np.any(close_to_neg):
                    wf = repulsion_strength / (dist_to_neg_wall[close_to_neg]**2 + 1e-10)
                    wf = np.minimum(wf, 100.0)
                    wall_force[close_to_neg] -= wf
                
                forces[:, k] += wall_force
            
            # Normalize forces to prevent large steps
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            # Update positions
            centers += step * forces
            centers = np.clip(centers, -0.499, 0.499)
            
            # Compute new radius
            new_r = get_min_radius(centers)
            
            # Adaptive step size
            if new_r > current_r:
                step = min(step * 1.1, 0.02)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.9, 0.001)
                no_improve_count += 1
            
            current_r = new_r
            
            # If stuck for too long, add random perturbation
            if no_improve_count > max_no_improve:
                n_perturb = max(1, N // 10)
                perturbed = np.random.choice(N, n_perturb, replace=False)
                centers[perturbed] += np.random.randn(n_perturb, 3) * 0.01
                centers = np.clip(centers, -0.499, 0.499)
                no_improve_count = 0
                step = 0.005
        
        return best_local_centers, best_local_r

    def greedy_optimize(centers, steps=20000):
        """Greedy single-pair hill climbing for local refinement."""
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step_size = 0.01
        
        for i in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            min_val = np.min(dist)
            if min_val < 1e-10:
                break
            
            idx_min = np.argmin(dist)
            i_idx, j_idx = np.unravel_index(idx_min, (N, N))
            
            # Check wall constraints
            wall_dists = 0.5 - np.abs(centers)
            min_wall_val = np.min(wall_dists)
            
            # Move the closest pair apart
            dir_ij = centers[i_idx] - centers[j_idx]
            norm_dir = np.linalg.norm(dir_ij)
            if norm_dir > 1e-12:
                dir_ij /= norm_dir
            else:
                dir_ij = np.random.randn(3)
                dir_ij /= np.linalg.norm(dir_ij)
                
            centers[i_idx] += dir_ij * step_size
            centers[j_idx] -= dir_ij * step_size
            
            # Handle walls
            for pt in [i_idx, j_idx]:
                for k in range(3):
                    if centers[pt, k] > 0.499:
                        centers[pt, k] = 0.499
                    elif centers[pt, k] < -0.499:
                        centers[pt, k] = -0.499
            
            new_r = get_min_radius(centers)
            
            if new_r > current_r:
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                # Revert if it got worse
                centers[i_idx] -= dir_ij * step_size
                centers[j_idx] += dir_ij * step_size
                for pt in [i_idx, j_idx]:
                    for k in range(3):
                        if centers[pt, k] > 0.499:
                            centers[pt, k] = 0.499
                        elif centers[pt, k] < -0.499:
                            centers[pt, k] = -0.499
            
            current_r = get_min_radius(centers)
            
            # Random jitter to escape local minima
            if i % 2000 == 0 and i > 0:
                pt_idx = np.random.randint(N)
                centers[pt_idx] += np.random.randn(3) * 0.01
                centers[pt_idx] = np.clip(centers[pt_idx], -0.499, 0.499)
                current_r = get_min_radius(centers)
        
        return best_local_centers, best_local_r

    # Initial Configurations
    
    # 1. Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    # 2. FCC-like Initialization
    s = 0.3
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
    
    dists_from_origin = np.linalg.norm(fcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_fcc = fcc_grid[order[:N]]
    
    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    # 4. BCC-like Initialization (Body-Centered Cubic)
    bcc_pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k]) * 0.35
                bcc_pts.append(base)
                bcc_pts.append(base + np.array([0.175, 0.175, 0.175]))
    bcc_grid = np.array(bcc_pts)
    bcc_grid -= bcc_grid.mean(axis=0)
    extent = np.max(np.abs(bcc_grid))
    if extent > 0.5:
        bcc_grid *= (0.5 / extent)
    dists_from_origin = np.linalg.norm(bcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_bcc = bcc_grid[order[:N]]
    
    # Run force-directed optimization on each initial configuration
    inits = [
        ("grid", init_grid),
        ("fcc", init_fcc),
        ("random", init_random),
        ("bcc", init_bcc)
    ]
    
    for name, init in inits:
        if time.time() - start_time > time_limit - 10:
            break
        c, r = optimize(init, max_steps=20000)
        if r > best_r:
            best_r = r
            best_centers = c
    
    # Refine the best with greedy optimization
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        c_greedy, r_greedy = greedy_optimize(best_centers, steps=15000)
        if r_greedy > best_r:
            best_r = r_greedy
            best_centers = c_greedy
    
    # Final force-directed polish
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        c_final, r_final = optimize(best_centers, max_steps=10000, step_size=0.002)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final
    
    if best_centers is None:
        return init_grid
    
    return best_centers
