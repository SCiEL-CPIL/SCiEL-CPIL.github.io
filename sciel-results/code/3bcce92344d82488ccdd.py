"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # --- Stage 1: Fibonacci Disk Initialization ---
    # This provides a uniform, non-lattice starting configuration that avoids
    # the periodic artifacts of ring-based layouts and the "clumping" of hex grids
    # when truncated. It ensures a good initial spread.
    
    n = N
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    # Radius distribution for uniform area coverage
    # r_i = R * sqrt(i / n)
    # We scale R slightly less than 1 to allow for expansion later
    R_init = 0.95
    indices = np.arange(1, n + 1)
    radii = R_init * np.sqrt(indices / n)
    angles = golden_angle * indices
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # --- Stage 2: Discrete Event-Driven Collision Dynamics ---
    # We treat the points as hard disks.
    # We want to maximize the minimum distance.
    # This is equivalent to letting the disks expand until they touch.
    # 
    # Algorithm:
    # 1. Start with a small target radius r (or just resolve overlaps based on current min dist).
    # 2. Better: Iterate by finding all pairs closer than a threshold, 
    #    and pushing them apart.
    # 3. To maximize the global min distance, we can use a "soft" repulsion force
    #    that is very strong for close pairs and decays rapidly.
    #    However, the directive asks for "resolving all pairwise overlaps by projecting them 
    #    onto the minimal separation manifold". This is a hard constraint solver step.
    #
    # Let's implement a hybrid: 
    # 1. Find current minimum distance d_min.
    # 2. Set target separation d_target = d_min * (1 + epsilon) or a fixed small increment.
    # 3. Identify all pairs with dist < d_target.
    # 4. Apply displacement vectors to separate them.
    # 5. Project to boundary.
    # 6. Repeat.
    #
    # To be efficient with N=300, we can vectorize the distance calculation.
    # Number of pairs is ~45k. This is manageable.
    
    # We will use a "Relaxation" loop.
    # We want to maximize r. 
    # Let's define a function to compute the score.
    
    def compute_score(pts):
        # Pairwise distances
        # Using broadcasting might be memory heavy for 300x300x2, but 300x300 is 90k floats, which is fine.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist)
        
        # Boundary distance
        norms = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        return r, min_pair_dist, min_bound_dist, dist

    # Initial score
    current_r, min_pd, min_bd, dist_mat = compute_score(points)
    best_r = current_r
    best_points = points.copy()
    
    # Relaxation parameters
    # We will increase a "target radius" r_target and try to push points apart to satisfy it.
    # If we can't satisfy it (system is too tight), we keep the best valid configuration found.
    # Actually, a better approach for hard disks is to just push apart any overlap.
    # But we want to maximize the radius.
    
    # Strategy: 
    # 1. Start with a small r_target.
    # 2. In each iteration, find all pairs with dist < 2*r_target.
    # 3. Push them apart by a small amount proportional to the overlap.
    # 4. Also push points inside the boundary out.
    # 5. If the system stabilizes (max displacement < threshold), increase r_target slightly.
    # 6. If it doesn't stabilize or time runs out, return best.
    
    r_target = 0.05  # Start slightly below expected optimum
    # The known optimum is ~0.05315. Let's start at 0.04 to be safe and expand.
    r_target = 0.04
    
    # Displacement strength
    # We want to resolve overlaps.
    # For a pair (i,j) with dist d < 2*r_target:
    # overlap = 2*r_target - d
    # Push vector = (overlap / 2) * unit_vec
    # This is a "relaxation" step. We might need to do multiple sub-steps or use a smaller factor.
    
    # To ensure we don't oscillate, we use a damping factor.
    damping = 0.5
    
    max_iterations = 5000
    min_displacement_threshold = 1e-6
    
    for iter_idx in range(max_iterations):
        if time.time() > deadline:
            break
            
        # 1. Compute distances
        # Reusing dist_mat from previous step is dangerous if we move points, so recompute.
        # Optimization: Only recompute if we moved points.
        
        # Find pairs closer than 2 * r_target
        # We need the full distance matrix for this check.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Find indices of close pairs
        # Using argwhere on a boolean mask
        close_mask = dist < 2.0 * r_target
        # Get indices
        i_idx, j_idx = np.nonzero(close_mask)
        
        # Exclude i == j (already inf, so shouldn't appear, but safe)
        valid_pairs = i_idx != j_idx
        i_idx = i_idx[valid_pairs]
        j_idx = j_idx[valid_pairs]
        
        # Symmetrize: if (i,j) is in list, (j,i) is also in list.
        # We only need to process each pair once to avoid double counting forces.
        # Keep only i < j
        unique_mask = i_idx < j_idx
        i_idx = i_idx[unique_mask]
        j_idx = j_idx[unique_mask]
        
        max_disp = 0.0
        
        if len(i_idx) > 0:
            # Calculate overlaps
            # d_ij = dist[i_idx, j_idx]
            d_ij = dist[i_idx, j_idx]
            overlaps = 2.0 * r_target - d_ij
            
            # Vectors from i to j
            vecs = points[j_idx] - points[i_idx]
            
            # Unit vectors
            norms = d_ij
            # Avoid division by zero (should be handled by mask, but safe)
            unit_vecs = vecs / norms[:, np.newaxis]
            
            # Displacement for each point in the pair
            # We push i away from j, and j away from i.
            # disp_i = -0.5 * overlap * unit_vec
            # disp_j = +0.5 * overlap * unit_vec
            
            disp_mag = 0.5 * overlaps * damping
            disp_i = -disp_mag[:, np.newaxis] * unit_vecs
            disp_j = disp_mag[:, np.newaxis] * unit_vecs
            
            # Apply displacements
            # Use np.add.at to handle multiple updates to same index
            np.add.at(points, i_idx, disp_i)
            np.add.at(points, j_idx, disp_j)
            
            max_disp = np.max(np.abs(disp_i))
            
        # 2. Boundary Constraint
        # If a point is outside (norm > 1 - r_target), push it in.
        # Actually, the constraint is norm <= 1 - r.
        # Let's enforce norm <= 1 - r_target - epsilon to be safe.
        # Or simply: if norm > 1 - r_target, scale it back.
        
        norms = np.linalg.norm(points, axis=1)
        limit = 1.0 - r_target
        outside_mask = norms > limit
        
        if np.any(outside_mask):
            # Scale points back to the boundary
            # We can project them radially.
            # To avoid just sticking to the boundary, we might want to push them in slightly.
            # But for hard disk packing, they should rest on the boundary.
            # Let's project exactly to limit.
            scale = limit / norms[outside_mask]
            points[outside_mask] *= scale[:, np.newaxis]
            
            # Calculate max displacement for convergence check
            # This is an approximation
            max_disp = max(max_disp, 0.01) # Assume significant change if boundary enforced
            
        # 3. Check Convergence / Update Target
        # If max displacement is very small, the system has reached a local equilibrium for this r_target.
        # We can then increase r_target.
        
        # Calculate current score to track best
        curr_r, curr_pd, curr_bd, _ = compute_score(points)
        
        if curr_r > best_r:
            best_r = curr_r
            best_points = points.copy()
            
        # Heuristic for increasing r_target:
        # If we made very little movement, we are stable. Increase r_target.
        # If we made large movement, keep r_target and continue relaxing.
        
        if max_disp < min_displacement_threshold:
            # Stable. Increase target radius.
            # How much? 
            # The gap between current min pair dist and 2*r_target is small.
            # We can increase r_target by a small step.
            # Step size could be adaptive.
            # Let's try a fixed small step initially, maybe increasing it.
            r_target += 0.0005
            # If r_target gets too high, we might get stuck. 
            # The max possible is ~0.053.
            if r_target > 0.06:
                # We've exceeded known bounds, likely stuck or optimal.
                # Stop increasing.
                pass
        
        # Additional check: If we are stuck in a bad local minimum, 
        # sometimes "jiggling" helps, but let's keep it simple first.
        
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
    # Final Score Calculation and Return
    # Ensure the returned points are valid.
    # The best_points might be from an earlier, lower r_target if the system became invalid or worse later?
    # compute_score handles validity implicitly by returning a low r if invalid (min dist < 0 or boundary < 0).
    # However, if points go outside, boundary dist is negative.
    # We track best_r, so we return best_points.
    
    # One last check: ensure best_points are strictly inside.
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    return best_points

# EVOLVE-BLOCK-END
