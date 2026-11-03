import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the exact packing radius for a set of points.
    Score = min(min_pair_dist / 2, min_boundary_dist)
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Calculate pairwise distances
    # For N=300, N^2 is 90,000, which is fast enough for final checks
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def fast_local_search(points, time_limit, K=10):
    """
    Fast Local Search using neighbor tracking.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Precompute initial neighbor lists
    # We store indices of K nearest neighbors for each point
    # Also store the distances to these neighbors
    neighbors = []
    neighbor_dists = []
    
    def compute_neighbors(idx):
        p = points[idx]
        # Distance to all others
        diffs = points - p
        dists = np.linalg.norm(diffs, axis=1)
        dists[idx] = np.inf
        # Get K nearest
        k_indices = np.argpartition(dists, K)[:K]
        # Sort them by distance
        k_dists = dists[k_indices]
        sort_idx = np.argsort(k_dists)
        return k_indices[sort_idx], k_dists[sort_idx]
    
    for i in range(n):
        idxs, ds = compute_neighbors(i)
        neighbors.append(idxs)
        neighbor_dists.append(ds)
        
    # Current minimum distance for each point (min of boundary dist and min neighbor dist)
    # boundary_dist[i] = 1 - norm(points[i])
    # min_neighbor_dist[i] = neighbor_dists[i][0]
    
    # We will greedily move points.
    # Strategy: Find the point with the smallest "bottleneck" (min of its constraints).
    # Move it to improve that constraint.
    
    # To speed up, we can just iterate through points and try to improve them.
    # A more sophisticated approach: maintain a priority queue of points by their current min-dist.
    # But for simplicity and speed in Python, we just loop.
    
    # Precompute current min-dist for each point
    min_dists = np.zeros(n)
    for i in range(n):
        b_dist = 1.0 - np.linalg.norm(points[i])
        n_dist = neighbor_dists[i][0] if K > 0 else 1.0
        min_dists[i] = min(b_dist, n_dist)
        
    # Sort points by min_dist (worst first)
    order = np.argsort(min_dists)
    
    # We will do multiple passes
    # In each pass, we try to improve the worst points
    
    # Step size for movement
    base_step = 0.01
    
    # Main loop
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate through points in order of their current weakness
        # We re-sort every few passes or just use a fixed order for speed?
        # Re-sorting is O(N log N) which is fast.
        # Let's just iterate 0..N-1 but check if time allows.
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
                
            p = points[i]
            b_dist = 1.0 - np.linalg.norm(p)
            
            # Get current neighbors
            n_idxs = neighbors[i]
            n_ds = neighbor_dists[i]
            min_n_dist = n_ds[0]
            
            current_min = min(b_dist, min_n_dist)
            
            # Try to move point i to increase current_min
            
            # Candidate directions:
            # 1. Away from the closest neighbor
            # 2. Away from the boundary (if boundary is limiting)
            # 3. Random small perturbations
            
            best_gain = 0
            best_new_pos = None
            
            # Determine the limiting constraint
            if b_dist < min_n_dist:
                # Boundary is limiting
                # Move radially outward
                if np.linalg.norm(p) > 1e-6:
                    dir = p / np.linalg.norm(p)
                else:
                    dir = np.random.normal(0, 1, 2)
                    dir /= np.linalg.norm(dir)
                
                # Try a few step sizes
                for step in [0.001, 0.005, 0.01, 0.02]:
                    new_p = p + dir * step
                    # Check if it's still valid (inside circle)
                    if np.linalg.norm(new_p) >= 1.0:
                        continue
                    
                    # Check against neighbors
                    # We only need to check the K neighbors + maybe a few more if we moved far
                    # For safety, check all neighbors in the list
                    min_new_dist = 1.0
                    for j in n_idxs:
                        d = np.linalg.norm(new_p - points[j])
                        if d < min_new_dist:
                            min_new_dist = d
                            if min_new_dist < current_min: # Early exit if it gets worse
                                break
                    
                    new_b_dist = 1.0 - np.linalg.norm(new_p)
                    new_min = min(new_b_dist, min_new_dist)
                    
                    if new_min > current_min + 1e-10:
                        gain = new_min - current_min
                        if gain > best_gain:
                            best_gain = gain
                            best_new_pos = new_p
            else:
                # Neighbor is limiting
                # Move away from the closest neighbor
                closest_idx = n_idxs[0]
                closest_p = points[closest_idx]
                diff = p - closest_p
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    dir = diff / dist
                else:
                    dir = np.random.normal(0, 1, 2)
                    dir /= np.linalg.norm(dir)
                
                # Also consider moving away from other close neighbors
                # We'll just try moving away from the closest one first
                
                for step in [0.001, 0.005, 0.01, 0.02]:
                    new_p = p + dir * step
                    
                    # Check boundary
                    new_b_dist = 1.0 - np.linalg.norm(new_p)
                    if new_b_dist < 0:
                        continue
                    if new_b_dist < current_min:
                        # If boundary becomes the new limit and it's worse, skip
                        # But we need to check if it's still better than current_min
                        pass 
                    
                    # Check against neighbors
                    min_new_dist = 1.0
                    for j in n_idxs:
                        d = np.linalg.norm(new_p - points[j])
                        if d < min_new_dist:
                            min_new_dist = d
                            if min_new_dist < current_min:
                                break
                            
                    new_min = min(new_b_dist, min_new_dist)
                    
                    if new_min > current_min + 1e-10:
                        gain = new_min - current_min
                        if gain > best_gain:
                            best_gain = gain
                            best_new_pos = new_p
            
            if best_new_pos is not None:
                points[i] = best_new_pos
                improved = True
                # Update neighbors for i and for its neighbors
                # This is the expensive part. 
                # We update i's neighbors.
                new_idxs, new_ds = compute_neighbors(i)
                neighbors[i] = new_idxs
                neighbor_dists[i] = new_ds
                
                # We should also update the neighbor lists of the points that are now neighbors of i
                # To keep it simple and fast, we might skip this or do it lazily.
                # If we skip, the neighbor lists become stale.
                # Stale neighbor lists might miss new close collisions.
                # However, since we are moving small steps, the set of K-nearest neighbors rarely changes drastically.
                # Let's update the neighbors of the points in new_idxs to be safe.
                for j in new_idxs:
                     # Only update if i is in j's neighbor list or close enough
                     # Actually, just recomputing j's neighbors is O(N) which is 300 ops.
                     # If we do this for every improvement, it might be slow.
                     # Let's limit updates.
                     pass # Skip for now to save time, rely on the "check against neighbors" in the move generation to catch issues.
                     # The "check against neighbors" uses the *old* neighbor list.
                     # This is a potential flaw: if a point moves into a collision with a non-neighbor, we might miss it in the move check.
                     # Mitigation: The move check is conservative. If it looks good against old neighbors, it's likely good.
                     # We will do a full validation at the end of the FLS phase.

        if not improved:
            # No improvements in this pass. 
            # We could try smaller steps or random kicks.
            # Let's do a small random kick on the worst point to escape local minima.
            worst_idx = order[0]
            p = points[worst_idx]
            kick = np.random.normal(0, 0.005, 2)
            new_p = p + kick
            if np.linalg.norm(new_p) < 1.0:
                 # Check if it's actually better? 
                 # Just apply it to break symmetry.
                 points[worst_idx] = new_p
                 # Update neighbors for worst_idx
                 new_idxs, new_ds = compute_neighbors(worst_idx)
                 neighbors[worst_idx] = new_idxs
                 neighbor_dists[worst_idx] = new_ds
                 
            # If we're stuck, we might just break and rely on polish.
            # But let's try a few more passes with smaller steps.
            base_step *= 0.9
            if base_step < 1e-4:
                break

    return points

def fast_gradient_polish(points, time_limit):
    """
    Gradient ascent on the minimum distance.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # We want to maximize r = min( min_{i<j} d_ij/2, min_i (1 - ||p_i||) )
    # This is equivalent to minimizing -r.
    # The gradient is non-smooth, but we can approximate it.
    # At any point, the "active" constraints are the pairs and boundaries that achieve the minimum.
    # We can compute the gradient of the sum of squared active constraints, or just use a simple heuristic.
    
    # A simpler and robust approach: 
    # Iterate:
    # 1. Find the current minimum distance r and the active set of constraints.
    # 2. Compute the gradient of the objective for these active constraints.
    # 3. Take a step in the gradient direction.
    
    # Since N=300, finding the global min and active set is O(N^2).
    # We can afford this if we don't do it too often.
    
    lr = 0.01 # Learning rate
    
    while time.time() - start_time < time_limit:
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        rad = np.linalg.norm(points, axis=1)
        b_dists = 1.0 - rad
        
        # Find global min
        min_pair = np.min(dists)
        min_bound = np.min(b_dists)
        r = min(min_pair / 2.0, min_bound)
        
        # Determine active constraints
        # Threshold for "active"
        eps = 1e-4
        active_pairs = np.argwhere(dists < min_pair + eps)
        active_bounds = np.argwhere(b_dists < min_bound + eps)
        
        # If no active constraints (shouldn't happen), break
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
            
        # Compute gradient
        # We want to maximize r.
        # Let's use a simple heuristic gradient:
        # For each active pair (i,j), push i and j apart.
        # For each active bound i, push i outwards.
        
        grad = np.zeros_like(points)
        
        # Pair contributions
        # We normalize the gradients to avoid large steps from many pairs
        count_pairs = len(active_pairs)
        if count_pairs > 0:
            for i, j in active_pairs:
                if i == j: continue
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist > 1e-6:
                    dir = d / dist
                    # Push i in dir, j in -dir
                    grad[i] += dir
                    grad[j] -= dir
        
        # Bound contributions
        count_bounds = len(active_bounds)
        if count_bounds > 0:
            for i in active_bounds:
                rad_i = np.linalg.norm(points[i])
                if rad_i > 1e-6:
                    dir = points[i] / rad_i
                    grad[i] += dir
                else:
                    # If at center, push randomly
                    grad[i] += np.random.normal(0, 1, 2)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad /= grad_norm
        else:
            break # No gradient direction
            
        # Take step
        # Adaptive step size
        step = lr * 0.01 # Small step
        
        new_points = points + grad * step
        
        # Project inside circle
        new_rad = np.linalg.norm(new_points, axis=1)
        mask = new_rad > 1.0
        if np.any(mask):
            new_points[mask] /= new_rad[mask, np.newaxis]
            
        # Check if score improved
        new_score = compute_score(new_points)
        if new_score > r:
            points = new_points
            # Increase lr slightly if improving
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease lr
            lr = max(lr * 0.9, 1e-5)
            if lr < 1e-5:
                break

    return points

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Density of hex packing is pi / (2 * sqrt(3)) * r^2 per area?
    # Area per circle in hex lattice with spacing s is s^2 * sqrt(3)/2.
    # Circle area is pi * r^2.
    # r = s/2.
    # Area per circle = s^2 * sqrt(3)/2 = (2r)^2 * sqrt(3)/2 = 2 * sqrt(3) * r^2.
    # Total area = pi * 1^2 = pi.
    # N * 2 * sqrt(3) * r^2 = pi
    # r^2 = pi / (2 * sqrt(3) * N)
    # r = sqrt(pi / (2 * sqrt(3) * N))
    # s = 2r
    
    r_est = math.sqrt(math.pi / (2 * math.sqrt(3) * N))
    s = 2 * r_est * 0.98 # Slight compression to ensure we fit
    
    y_step = s * math.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s / 2
        for i in range(-max_i, max_i + 1):
            x = i * s + offset
            if math.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        # Keep the ones closest to the center? Or random?
        # Closest to center is better for boundary constraint.
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Fast Local Search
    # Allocate 80% of time to FLS
    fls_time = 110.0
    fls_points = fast_local_search(best_points, fls_time)
    fls_score = compute_score(fls_points)
    
    if fls_score > best_score:
        best_score = fls_score
        best_points = fls_points
        
    # 3. Fast Gradient Polish
    # Allocate remaining time (minus safety) to Polish
    remaining_time = time_limit - (time.time() - start_time) - 5.0
    if remaining_time > 5.0:
        polish_points = fast_gradient_polish(best_points, remaining_time)
        polish_score = compute_score(polish_points)
        
        if polish_score > best_score:
            best_score = polish_score
            best_points = polish_points

    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        # Fallback to a valid configuration
        # Return the initial lattice if it was valid
        if np.all(np.linalg.norm(points, axis=1) <= 1.0):
            return points
        return np.zeros((N, 2))
        
    return best_points

