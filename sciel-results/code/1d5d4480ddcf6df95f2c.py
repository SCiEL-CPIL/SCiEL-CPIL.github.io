"""Pack 50 equal circles in a unit square using Projected Gradient Ascent with Dynamic Target Radius."""
import numpy as np
import time

N = 50
DEADLINE = 148.0  # Safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _project(centers):
    """Project centers onto the square [-0.5, 0.5]^2."""
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    return centers

def _resolve_overlaps(centers, target_r, max_iters=1000, tol=1e-6):
    """
    Iteratively resolve overlaps for a given target radius.
    Returns: (centers, converged, min_dist)
    """
    centers = centers.copy()
    step_size = 0.5
    decay = 0.99
    
    for it in range(max_iters):
        centers = _project(centers)
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Check if all pairs are separated by at least 2*target_r
        # We only care about pairs where dist < 2*target_r
        # Note: dist[i,i] = 0, so we ignore diagonal
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        if min_dist >= 2 * target_r - tol:
            return centers, True, min_dist
            
        # Calculate forces (gradients of separation)
        # For each pair i, j with dist < 2*target_r, push them apart
        forces = np.zeros_like(centers)
        
        # Vectorized force calculation
        # diff[i,j] = c_i - c_j
        # We want to move c_i away from c_j if they are too close.
        # Direction: (c_i - c_j) / dist
        # Magnitude: proportional to overlap amount? Or just fixed step?
        # Standard approach: move by 0.5 * (2r - dist) * direction
        
        # Only compute for close pairs to save time? 
        # For N=50, 50x50 matrix is small, so vectorize fully.
        
        # Mask for overlapping pairs
        mask = dist < 2 * target_r
        # Exclude diagonal
        mask[np.diag_indices_from(mask)] = False
        
        if not np.any(mask):
            return centers, True, min_dist

        # For each overlapping pair (i, j), add force to i and -force to j
        # diff[i,j] is vector from j to i.
        # We want to increase dist.
        # Force on i: direction * magnitude
        # Force on j: -direction * magnitude
        
        # To avoid double counting and ensure symmetry, 
        # let's sum forces properly.
        # F_i = sum_{j != i, mask[i,j]} (c_i - c_j) / dist[i,j] * step
        
        # We can do this with matrix multiplication or loops.
        # Loops are fine for N=50.
        
        for i in range(N):
            for j in range(N):
                if mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-9:
                        # If coincident, push randomly or along axis
                        # This is rare if we start well
                        dir_vec = np.random.randn(2)
                        dir_vec /= np.linalg.norm(dir_vec)
                    else:
                        dir_vec = (centers[i] - centers[j]) / d
                    
                    # Overlap amount
                    overlap = 2 * target_r - d
                    # Move half the overlap towards each other
                    # Actually, if we move i by 0.5*overlap and j by -0.5*overlap, dist increases by overlap.
                    # But we have many interactions. A simple relaxation is better.
                    # Let's use a fixed step size for stability.
                    # Or proportional to overlap.
                    # Let's try proportional to overlap, scaled by step_size.
                    
                    force = dir_vec * (overlap * step_size)
                    forces[i] += force
                    # forces[j] -= force # Handled when i=j, j=i iteration? 
                    # Wait, if I loop i, j over all pairs, I will double count.
                    # Let's only loop i < j.
                    pass
        
        # Let's restart the force calculation with i < j to be clean
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-9:
                        dir_vec = np.random.randn(2)
                        dir_vec /= np.linalg.norm(dir_vec)
                    else:
                        dir_vec = (centers[i] - centers[j]) / d
                    
                    overlap = 2 * target_r - d
                    # Move i and j apart
                    # Step size controls convergence speed
                    move = dir_vec * (overlap * step_size)
                    forces[i] += move
                    forces[j] -= move
        
        centers += forces
        step_size *= decay
        
        # Check convergence: if max movement is small
        # Actually, the min_dist check above is the primary convergence check.
        # If we don't converge in max_iters, we return the best effort.
        
    return centers, False, min_dist

def _make_hexagonal_grid():
    """Create a good initial hexagonal packing."""
    # Hexagonal packing density is higher than square.
    # We need to fit 50 circles.
    # Let's generate a hexagonal lattice and select the best 50.
    
    # Row spacing: sqrt(3)/2 * side
    # Col spacing: side
    # Let's try to fit a grid that covers the square.
    
    points = []
    # Try different scales
    for scale in [0.15, 0.12, 0.10, 0.08, 0.06]:
        # Hex grid
        # y = row * sqrt(3)/2 * scale
        # x = col * scale + (row % 2) * scale/2
        
        max_row = int(1.0 / (scale * np.sqrt(3) / 2)) + 2
        max_col = int(1.0 / scale) + 2
        
        for row in range(-max_row, max_row + 1):
            for col in range(-max_col, max_col + 1):
                x = col * scale + (row % 2) * scale / 2
                y = row * scale * np.sqrt(3) / 2
                # Center around 0
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        
        if len(points) >= N:
            # We have enough points. Now we need to select 50 that are well-distributed.
            # Simple heuristic: take the first N? No, they might be clustered.
            # Better: Randomly sample from the valid grid points?
            # Or just use all points and let the optimizer figure it out?
            # The optimizer works on fixed N.
            # Let's just take a random subset of N points from the valid grid.
            np.random.shuffle(points)
            return np.array(points[:N])
    
    # Fallback: Square grid
    ticks = np.linspace(-0.45, 0.45, 8)
    return np.array([(x, y) for y in ticks for x in ticks])[:N]

def _make_random_grid():
    """Random initial positions."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    inits = [
        _make_hexagonal_grid(),
        _make_random_grid(),
        _make_random_grid()
    ]
    
    # Dynamic Target Radius Strategy
    # Start with a small target radius, increase it if we can pack, else decrease.
    # This is a "bisection-like" or "expansion" search.
    
    # We will run multiple iterations of:
    # 1. Try to pack with target_r
    # 2. If successful (converged), calculate actual radius.
    # 3. If actual radius > target_r, we can try a higher target.
    # 4. If failed, try a lower target.
    
    # Let's manage a list of "promising" target radii.
    # Or simpler:
    # For each init:
    #   Start target_r = 0.06
    #   Loop:
    #     Resolve overlaps for target_r
    #     If converged:
    #       r_actual = _radius(centers)
    #       if r_actual > best_r: update best
    #       # Increase target_r slightly to push for better packing
    #       target_r *= 1.05
    #     Else:
    #       # Failed to pack, maybe target was too high or local min.
    #       # Try slightly lower target to get a valid config, then refine?
    #       # Or just break and try next init.
    #       # Actually, if it fails, the centers are still somewhat separated.
    #       # We can use the current centers as a start for a lower target.
    #       target_r *= 0.95
    #       if target_r < 0.05: break
    #     Check time
    
    # Better Strategy:
    # 1. Get a valid packing with a modest r (e.g., 0.06) using the resolver.
    # 2. Use that as a starting point for a "radius expansion" loop.
    #    In each step, increase r_target by a small delta.
    #    Run the resolver with the new r_target.
    #    If it converges, keep the new centers.
    #    If it fails, revert to previous centers and try a smaller delta or stop.
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Step 1: Establish a baseline packing
        # Try to pack with r=0.06
        current_centers = init_centers.copy()
        r_target = 0.06
        
        # Ensure we have a valid start
        success, resolved_centers, min_d = _resolve_overlaps(current_centers, r_target, max_iters=500)
        if success:
            current_centers = resolved_centers
            current_r = _radius(current_centers)
        else:
            # If even 0.06 fails, try 0.05
            r_target = 0.05
            success, resolved_centers, min_d = _resolve_overlaps(current_centers, r_target, max_iters=500)
            if success:
                current_centers = resolved_centers
                current_r = _radius(current_centers)
            else:
                # Skip this init if it's too bad
                continue
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current_centers.copy()
            
        # Step 2: Expand radius
        # Incrementally increase r_target
        # Delta starts small, increases if success, decreases if fail
        delta = 0.001
        last_success_r = current_r
        
        while time.time() - start_time < DEADLINE - 5:
            new_target = last_success_r + delta
            
            # Run resolver with new target
            # Use current_centers as start
            # Limit iterations to save time
            iters_allowed = 500
            if time.time() - start_time > DEADLINE - 10:
                break
                
            success, new_centers, min_d = _resolve_overlaps(current_centers, new_target, max_iters=iters_allowed)
            
            if success:
                new_r = _radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                
                # Success! Update state
                current_centers = new_centers
                last_success_r = new_r
                # Increase delta slightly to explore faster?
                # But be careful not to jump over the optimum.
                # Keep delta constant or slightly increase.
                delta *= 1.02 
                
                # If we are getting very close to the theoretical max or time is running out,
                # we might want to switch to a finer optimization.
                # For now, continue expanding.
                
            else:
                # Failure
                # Decrease delta
                delta *= 0.8
                if delta < 1e-5:
                    # We've converged as much as this method allows
                    break
                
                # Optionally, try to "repair" the failed configuration by
                # running the resolver with the *previous* successful target
                # but with more iterations? No, that's wasted time.
                # Just try a smaller step.
                
                # If we fail consistently, we might be stuck in a local minimum
                # that doesn't allow expansion.
                # We could try a random perturbation of current_centers and restart expansion.
                # Let's add a "kick" mechanism.
                # Perturb slightly
                current_centers += np.random.randn(N, 2) * 0.005
                current_centers = _project(current_centers)
                
                # Try again with the same delta? Or smaller?
                # Let's try smaller delta to be safe.
                pass

    # Final Polish:
    # Use a local optimizer on the best centers found to squeeze out more radius.
    # We can use a simple gradient ascent on the soft-min radius or just more expansion steps.
    # Since we have time, let's do a few more expansion steps with very small deltas.
    
    if best_centers is not None and time.time() - start_time < DEADLINE - 2:
        current_centers = best_centers.copy()
        current_r = best_r
        delta = 0.0001
        
        for _ in range(50): # Max 50 tiny steps
            if time.time() - start_time > DEADLINE - 1:
                break
                
            new_target = current_r + delta
            success, new_centers, _ = _resolve_overlaps(current_centers, new_target, max_iters=200)
            
            if success:
                new_r = _radius(new_centers)
                if new_r > current_r:
                    current_centers = new_centers
                    current_r = new_r
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = current_centers.copy()
            else:
                delta *= 0.9
                if delta < 1e-6:
                    break

    if best_centers is None:
        # Fallback
        ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
        ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
        best_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]

    return best_centers
