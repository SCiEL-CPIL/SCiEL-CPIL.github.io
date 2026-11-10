import numpy as np
import time

N = 100

def _radius_score(c):
    """Compute the exact packing radius for a configuration."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _force_model(c, r_target, alpha=50.0, beta=10.0):
    """
    Compute repulsive forces based on a soft constraint violation model.
    We want to push spheres apart if d_ij < 2*r_target and away from walls if dist < r_target.
    
    Force magnitude is proportional to the violation, sharpened by alpha.
    """
    c = np.asarray(c, dtype=np.float64)
    forces = np.zeros_like(c)
    
    # Pairwise repulsion
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Constraint: dists should be >= 2*r_target
    # Violation: max(0, 2*r_target - dists)
    # We use a smooth approximation: (2*r_target - dists) * sigmoid(alpha * (2*r_target - dists))
    # Or simpler: just use linear repulsion for overlapping/near-overlapping spheres
    # F_ij = k * (2*r_target - d_ij) * (c_i - c_j) / d_ij  if d_ij < 2*r_target
    
    # Vectorized calculation
    # Only consider pairs where dist < 2*r_target + margin
    margin = 0.05
    mask = dists < (2.0 * r_target + margin)
    mask = mask & (dists > 1e-6)
    
    # Get indices
    i_idx, j_idx = np.where(mask)
    
    if len(i_idx) > 0:
        d_vec = diff[i_idx, j_idx]
        d_val = dists[i_idx, j_idx]
        
        # Strength: linear in violation
        violation = 2.0 * r_target - d_val
        # Only repel if violation > 0, else attract slightly? No, just repel if close.
        # Actually, to maximize r, we want to push apart if d < 2r.
        # If d > 2r, force is 0.
        # Let's use a smooth potential: U = 0.5 * k * max(0, 2r - d)^2
        # F = -dU/dc = k * max(0, 2r - d) * (c_i - c_j)/d
        
        k = 1.0
        f_mag = k * np.maximum(0.0, violation)
        # Normalize direction
        f_vec = f_mag[:, np.newaxis] * (d_vec / d_val[:, np.newaxis])
        
        # Accumulate forces
        for idx in range(len(i_idx)):
            i = i_idx[idx]
            j = j_idx[idx]
            forces[i] += f_vec[idx]
            forces[j] -= f_vec[idx]
            
    # Wall repulsion
    # Constraint: |c_ik| + r_target <= 0.5  =>  0.5 - |c_ik| >= r_target
    # Violation: max(0, r_target - (0.5 - |c_ik|))
    wall_dist = 0.5 - np.abs(c)
    violation_w = r_target - wall_dist
    # Force pushes inward
    # F_ik = k * max(0, violation_w) * sign(c_ik) ? No, away from wall means towards center.
    # If c_ik > 0, wall is at 0.5. Force should be negative (towards 0).
    # If c_ik < 0, wall is at -0.5. Force should be positive (towards 0).
    # Direction = -sign(c_ik)
    dir_w = -np.sign(c)
    # Handle c=0 carefully, sign is 0, but we can just use 0 or random. 
    # If c=0, it's far from wall, so violation is likely 0.
    f_mag_w = k * np.maximum(0.0, violation_w)
    forces += f_mag_w[:, np.newaxis] * dir_w
    
    return forces

def _relax_local(c, r_target, max_iter=500, tol=1e-8):
    """
    Local relaxation: iteratively move spheres to satisfy constraints for a fixed r_target.
    If we can satisfy constraints for r_target, we can try a larger r_target.
    """
    c = c.copy()
    step_size = 0.01
    
    for it in range(max_iter):
        forces = _force_model(c, r_target)
        
        # Normalize forces to prevent explosion
        max_f = np.max(np.linalg.norm(forces, axis=1))
        if max_f > 1e-6:
            forces = forces / max_f * 0.1 # Scale down significantly for stability
        
        # Move
        c_new = c + step_size * forces
        
        # Project to cube
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Check if we improved the minimum distance / wall distance
        r_new = _radius_score(c_new)
        
        if r_new >= r_target - tol:
            c = c_new
        else:
            # If we violated, try smaller step or revert
            # Simple backtracking
            c_new = c + 0.5 * step_size * forces
            c_new = np.clip(c_new, -0.5, 0.5)
            if _radius_score(c_new) >= r_target - tol:
                c = c_new
            else:
                # Give up on this step, maybe jitter
                break
                
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Phase 1: Multi-start initialization
    # Use a combination of lattice and random starts
    
    rng = np.random.default_rng(42)
    
    starts = []
    
    # Lattice start
    s = 0.18
    n_dim = int(np.ceil(1.0 / s)) + 1
    coords = np.arange(-n_dim // 2, n_dim // 2 + 1) * s
    grid = np.array([(x, y, z) for z in coords for y in coords for x in coords])
    inside_mask = np.all(np.abs(grid) <= 0.5 - 1e-6, axis=1)
    inside_grid = grid[inside_mask]
    if len(inside_grid) >= N:
        dists_center = np.linalg.norm(inside_grid, axis=1)
        sorted_idx = np.argsort(dists_center)
        centers = inside_grid[sorted_idx[:N]].copy()
        centers += rng.uniform(-0.002, 0.002, size=(N, 3))
        centers = np.clip(centers, -0.5, 0.5)
        starts.append(centers)
        
    # Random starts
    for _ in range(3):
        c = rng.uniform(-0.45, 0.45, size=(N, 3))
        starts.append(c)
        
    # Process each start
    for c0 in starts:
        if time.time() > deadline - 10:
            break
            
        c = c0.copy()
        r = _radius_score(c)
        
        # Phase 2: Force-directed relaxation with increasing target radius
        # We try to push r up gradually
        r_target = r * 1.01
        max_r_attempts = 200
        stall_count = 0
        
        for attempt in range(max_r_attempts):
            if time.time() > deadline - 5:
                break
                
            # Try to relax to r_target
            c_relaxed = _relax_local(c, r_target, max_iter=100)
            
            r_new = _radius_score(c_relaxed)
            
            if r_new >= r_target - 1e-7:
                # Success! We achieved r_target.
                c = c_relaxed
                r = r_new
                if r > best_radius:
                    best_radius = r
                    best_centers = c.copy()
                
                # Increase target
                r_target = r * 1.005 + 0.0001
                stall_count = 0
            else:
                # Failed to reach r_target
                # Maybe we can still improve r a bit even if we didn't hit target?
                if r_new > r + 1e-7:
                    c = c_relaxed
                    r = r_new
                    if r > best_radius:
                        best_radius = r
                        best_centers = c.copy()
                    # Slightly increase target to continue
                    r_target = max(r_target, r * 1.001 + 0.00005)
                else:
                    # Stagnant
                    stall_count += 1
                    if stall_count > 5:
                        # Jitter and try again
                        c = c + rng.normal(0, 0.005, size=(N, 3))
                        c = np.clip(c, -0.5, 0.5)
                        r = _radius_score(c)
                        r_target = r * 1.002
                        stall_count = 0
                    else:
                        # Try smaller target increment
                        r_target = r * 1.001 + 0.00001
                        
    # Phase 3: Final local refinement (Geometric Jitter-Relax)
    # Focus on the tightest constraints
    if best_centers is not None:
        c = best_centers.copy()
        r = _radius_score(c)
        
        # Identify tight constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        wall_dists = 0.5 - np.abs(c)
        
        # Sort pairs by distance
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        sorted_pairs = np.argsort(pair_dists)
        
        # Try to optimize the tightest 10 pairs
        for idx in sorted_pairs[:10]:
            if time.time() > deadline - 2:
                break
                
            i, j = iu[0][idx], iu[1][idx]
            d_ij = dists[i, j]
            
            # If this pair is the bottleneck (d_ij/2 <= r + eps)
            if d_ij / 2.0 <= r + 1e-6:
                # Try to move i and j apart
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    dir_vec = vec / norm
                    # Try small displacements
                    for step in [0.001, 0.0005, 0.0002]:
                        c_trial = c.copy()
                        c_trial[i] += step * dir_vec
                        c_trial[j] -= step * dir_vec
                        c_trial = np.clip(c_trial, -0.5, 0.5)
                        r_trial = _radius_score(c_trial)
                        if r_trial > r + 1e-9:
                            c = c_trial
                            r = r_trial
                            if r > best_radius:
                                best_radius = r
                                best_centers = c.copy()
                            break
        
    # Final validation
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        best_centers = np.clip(rng.uniform(-0.4, 0.4, size=(N, 3)), -0.5, 0.5)
    
    return best_centers
