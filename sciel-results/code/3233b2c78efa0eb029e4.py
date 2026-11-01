"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _initial_config(n):
    """Generate a dense initial configuration using a 4x4x4 grid."""
    # 4x4x4 grid = 64 points. We need 50.
    # Coordinates in [-0.5, 0.5]
    # Spacing is 1/3. Points at -0.5, -0.166, 0.166, 0.5
    # This puts points exactly on the walls, which is good for maximizing radius.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64 points. We want 50.
    # Removing points from the corners/edges might be better? 
    # Actually, for a dense packing, we want to keep the structure.
    # Let's just take the first 50. The grid is symmetric.
    # To break symmetry slightly for optimization, we can add tiny noise.
    return grid[:n].copy()


def _relax_repulsion(c, r, max_iters=1000, tol=1e-8):
    """
    Relax positions to minimize overlap energy for a given radius r.
    Energy E = sum_{i<j} max(0, 2r - dist_ij)^2 + sum_i sum_k max(0, r - wall_dist_ik)^2
    We use gradient descent on the positions.
    """
    n = len(c)
    c = c.copy()
    
    # Precompute indices for pairs
    # This is fixed for N=50, but we can just loop.
    # For speed, we can vectorize the gradient calculation.
    
    for it in range(max_iters):
        # Calculate forces (negative gradient of energy)
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1e-10) # Avoid div by zero
        
        # Overlap amount: max(0, 2r - dist)
        overlap = np.maximum(0, 2*r - dists)
        # Gradient of (overlap)^2 w.r.t c_i is -2 * overlap * (diff_ij / dist_ij)
        # Force on i from j: 2 * overlap_ij * (c_i - c_j) / dist_ij
        # Note: diff[i,j] = c_i - c_j
        # Term: 2 * overlap[i,j] * diff[i,j] / dists[i,j]
        
        # Vectorized computation of forces from pairs
        # mask for i != j
        mask = ~np.eye(n, dtype=bool)
        
        # Avoid division by zero where dist is 0 (shouldn't happen with 1e-10)
        # Force contribution from pair (i,j) on i is 2 * overlap[i,j] * (c_i - c_j) / dist[i,j]
        # We sum over j for each i.
        
        # Create a matrix of unit vectors scaled by overlap
        # unit_vec[i,j] = diff[i,j] / dists[i,j]
        unit_vec = diff / dists[:, :, np.newaxis]
        
        # Weighted by 2 * overlap
        weighted_unit = 2 * overlap[:, :, np.newaxis] * unit_vec
        
        # Sum over j (axis 1) to get force on i
        pair_forces = np.sum(weighted_unit, axis=1)
        
        # Wall repulsion
        # Wall distance d_w = 0.5 - |c_ik|
        # Overlap w = max(0, r - d_w)
        # Energy term: w^2
        # Gradient w.r.t c_ik: -2 * w * sign(c_ik)
        # Force: 2 * w * sign(c_ik)
        
        wall_dists = 0.5 - np.abs(c)
        wall_overlap = np.maximum(0, r - wall_dists)
        # Force on c_ik is 2 * wall_overlap_ik * sign(c_ik)
        # If c_ik is 0, sign is 0, force is 0. This is a singularity but rare.
        # We can use np.sign.
        wall_forces = 2 * wall_overlap * np.sign(c)
        
        forces = pair_forces + wall_forces
        
        # Update positions
        # Simple gradient ascent on "free space" / descent on "energy"
        # We want to move in direction of forces (away from overlap)
        # Step size needs to be small enough to not oscillate.
        # Adaptive step size?
        step_size = 0.01
        
        # Normalize forces to prevent huge jumps?
        # Or just use a fixed small step.
        
        new_c = c + step_size * forces
        
        # Clip to cube
        new_c = np.clip(new_c, -0.5, 0.5)
        
        # Check convergence
        delta = np.linalg.norm(new_c - c)
        c = new_c
        
        if delta < tol:
            break
            
    return c


def _local_refine(c, max_iters=500, time_limit=20.0):
    """
    Local refinement using coordinate descent / Nelder-Mead style.
    Maximize r(c) directly.
    """
    t_start = time.time()
    c = c.copy()
    best_r = _compute_radius(c)
    
    # Simple coordinate descent with line search
    # For each point i, and each dimension d, try to move c[i,d] to maximize r.
    
    for it in range(max_iters):
        if time.time() - t_start > time_limit:
            break
            
        improved = False
        for i in range(N):
            for d in range(3):
                # Try small steps in + and - directions
                best_step = 0
                best_r_local = best_r
                
                # Current pos
                old_val = c[i, d]
                
                # Try steps
                # We want to increase the minimum distance.
                # The minimum distance is likely determined by the nearest neighbor or wall.
                # We can just try a few steps.
                for step in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                    new_val = old_val + step
                    if new_val < -0.5 or new_val > 0.5:
                        continue
                    
                    c[i, d] = new_val
                    r_try = _compute_radius(c)
                    
                    if r_try > best_r_local:
                        best_r_local = r_try
                        best_step = step
                
                if best_step != 0:
                    c[i, d] = old_val + best_step
                    best_r = best_r_local
                    improved = True
        
        if not improved:
            break
            
    return c, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    np.random.seed(42)
    
    # 1. Initial Configuration
    c = _initial_config(N)
    # Add small noise to break symmetry
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    c = np.clip(c, -0.499, 0.499)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # 2. Repulsion-based relaxation to increase radius
    # Start with a radius slightly smaller than current best
    r = best_r * 0.9
    
    # Iterate: relax for r, then increase r
    # We want to find the max r such that we can relax to E=0.
    
    # First, relax at current r to ensure it's valid
    c = _relax_repulsion(c, r, max_iters=200)
    c = np.clip(c, -0.499, 0.499)
    r_current = _compute_radius(c)
    
    if r_current > best_r:
        best_r = r_current
        best_c = c.copy()
    
    # Now try to increase r
    # Step size for r increase
    dr = 0.001
    max_r_attempts = 5000
    
    for i in range(max_r_attempts):
        if time.time() > deadline - 20.0:
            break
            
        # Try to increase r
        r_new = r_current + dr
        
        # Relax positions to fit r_new
        c_trial = _relax_repulsion(c, r_new, max_iters=50)
        c_trial = np.clip(c_trial, -0.499, 0.499)
        
        # Check actual radius achieved
        r_achieved = _compute_radius(c_trial)
        
        if r_achieved >= r_new - 1e-6:
            # Success, we can accept this new radius
            r_current = r_achieved
            c = c_trial
            if r_current > best_r:
                best_r = r_current
                best_c = c.copy()
        else:
            # Failed to fit r_new. 
            # Maybe we can still improve current r by relaxing more?
            # Or decrease r slightly and try again?
            # If we fail, it means the configuration is "tight" at r_current.
            # We can try to squeeze out more by local refinement on c at r_current.
            
            # Try local refinement on the current best configuration
            if time.time() < deadline - 10.0:
                c_refined, r_refined = _local_refine(best_c, max_iters=100, time_limit=5.0)
                if r_refined > best_r:
                    best_r = r_refined
                    best_c = c_refined
                    r_current = r_refined
                    c = c_refined.copy()
            
            # If local refinement didn't help, break the repulsion loop?
            # Or continue with smaller dr?
            # Let's try smaller dr
            dr *= 0.9
            if dr < 1e-5:
                break
            continue
            
    # 3. Final local refinement
    if time.time() < deadline - 5.0:
        c_final, r_final = _local_refine(best_c, max_iters=500, time_limit=deadline - time.time() - 2.0)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
