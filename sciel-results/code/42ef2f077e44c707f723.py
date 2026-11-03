"""
Strategy: Combinatorial Local Search (Best-Improvement Hill Climbing) with Cell-Based Neighborhoods.

Unlike the smooth surrogate approach which uses gradients to navigate the continuous space,
this approach treats the problem as a discrete optimization task. It identifies the "bottleneck"
spheres that limit the current radius and attempts to move them to better positions using a
coarse-to-fine local search. This allows the algorithm to make discrete jumps that can escape
the local minima where gradient-based methods often get stuck.

Key components:
1. Exact radius calculation.
2. Identification of the critical set (spheres involved in the minimum distance).
3. Local search for a single sphere within its "available space" (defined by current r).
4. Iterative improvement with random restarts.
"""
import numpy as np
import time
import itertools

N = 100
CUBE_HALF = 0.5

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _local_search_sphere(i, points, r_current, time_limit):
    """
    Attempt to improve the position of sphere i to allow for a larger radius.
    We search in a local neighborhood around points[i].
    The search is constrained by the current radius r_current to ensure we don't
    immediately create overlaps, but we aim to find a spot that increases the
    minimum distance to neighbors or walls.
    """
    p_i = points[i].copy()
    
    # Determine the "bottleneck" distance for sphere i
    # 1. Distance to walls
    wall_margin_i = CUBE_HALF - np.abs(p_i).max()
    # 2. Minimum distance to other spheres
    dists_to_others = np.linalg.norm(points - p_i, axis=1)
    dists_to_others[i] = np.inf
    min_dist_to_others = dists_to_others.min()
    
    # The current effective radius is limited by min(wall_margin_i, min_dist_to_others/2)
    # We want to move p_i to a location that maximizes this local minimum.
    
    # Define a search box around p_i
    # The box size should be related to the current spacing.
    # If we move too far, we might collide with others.
    # Let's try a box of size 0.5 * r_current in each dimension? Or maybe larger?
    # Actually, we can move up to r_current/2 away before definitely colliding with neighbors at distance r_current.
    # But we want to explore a bit more.
    
    search_radius = 0.6 * r_current
    if search_radius < 0.001:
        search_radius = 0.001
        
    # Coarse grid search
    steps = 10
    step_size = 2 * search_radius / steps
    
    best_pos = p_i.copy()
    best_local_score = -1.0
    
    # Generate grid points
    # To save time, we only search in the box
    # We can optimize by checking only points that are "promising"
    
    # Pre-compute distances to others for the current position to use as a heuristic?
    # For now, brute force the local grid.
    
    # We will iterate over a 3D grid
    # To keep it fast, we limit the number of points.
    # 10^3 = 1000 points. For each point, checking N-1 distances is ~100 ops.
    # 1000 * 100 = 100,000 ops. Very fast.
    
    # Let's refine the grid around the current position.
    # We can bias the grid towards directions that are "free".
    
    # Simple approach: Uniform grid in the box.
    xs = np.linspace(-search_radius, search_radius, steps)
    ys = np.linspace(-search_radius, search_radius, steps)
    zs = np.linspace(-search_radius, search_radius, steps)
    
    # We can reduce the number of points by using a smaller step if steps is large,
    # but 10 is fine.
    
    # Iterate
    # We can vectorize the evaluation of the grid points against the other spheres.
    
    # Create all candidate points
    # Grid shape: (steps, steps, steps, 3)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    candidates = np.stack([
        p_i[0] + gx.flatten(),
        p_i[1] + gy.flatten(),
        p_i[2] + gz.flatten()
    ], axis=1)
    
    # Filter candidates that are inside the cube
    mask = np.all(np.abs(candidates) <= CUBE_HALF, axis=1)
    candidates = candidates[mask]
    
    if len(candidates) == 0:
        return p_i, -1.0
        
    # Evaluate each candidate
    # For each candidate c, compute:
    # 1. Wall margin: CUBE_HALF - max(|c|)
    # 2. Min distance to other spheres / 2
    
    # Vectorized distance calculation
    # candidates: (M, 3)
    # points: (N, 3)
    # We need dist(c, p_j) for j != i
    
    # To save memory and time, we can do this in chunks if M is large, but M <= 1000 is small.
    # diff = candidates[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # dists[:, i] = np.inf
    # min_dists = dists.min(axis=1)
    
    # This creates a (M, N) array. 1000 * 100 = 100,000 floats. Fine.
    
    diff = candidates[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    dists[:, i] = np.inf # Exclude self
    
    min_dists = dists.min(axis=1)
    wall_margins = CUBE_HALF - np.abs(candidates).max(axis=1)
    
    # The new radius if we move to candidate k is min(wall_margins[k], min_dists[k]/2)
    # BUT, we also have to ensure that this move doesn't decrease the global radius
    # below the current r_current? No, we want to maximize the global radius.
    # If we move sphere i, the new global radius is min(
    #   min_{k != i} (local radius of k), 
    #   new local radius of i
    # )
    # The local radii of other spheres k might change because their distance to i changed.
    # So we can't just look at i's local radius.
    
    # However, a common heuristic in local search is:
    # Only accept a move if it increases the *minimum* distance involving i,
    # AND does not decrease the distances of other spheres to i below their current minimums.
    
    # Actually, the strict condition is:
    # new_global_r = min(
    #   min_{j<k, j!=i, k!=i} dist(j,k)/2,  # Unchanged pairs
    #   min_{j!=i} dist(j, new_i)/2,         # New pairs involving i
    #   new_wall_margin_i
    # )
    
    # We know current_global_r = r_current.
    # Unchanged pairs have min dist >= 2*r_current.
    # So new_global_r = min(r_current, min_{j!=i} dist(j, new_i)/2, new_wall_margin_i)
    # Wait, this is only true if the *other* spheres' minimum distances to *each other* are >= 2*r_current.
    # And they are, by definition of r_current.
    # BUT, the minimum distance of sphere j to *other* spheres (excluding i) is >= 2*r_current.
    # The minimum distance of sphere j to i might be < 2*r_current in the new config?
    # No, we are calculating min_{j!=i} dist(j, new_i).
    
    # So the new global radius is indeed:
    # new_r = min(r_current, min_{j!=i} dist(j, new_i)/2, new_wall_margin_i)
    
    # This simplifies things! We just need to find a candidate that maximizes:
    # min(min_{j!=i} dist(j, new_i)/2, new_wall_margin_i)
    # subject to this value being > r_current (to improve).
    
    # Actually, we want to maximize this value. Let's call it potential_r.
    potential_r = np.minimum(min_dists / 2.0, wall_margins)
    
    # Find the best candidate
    best_idx = np.argmax(potential_r)
    best_val = potential_r[best_idx]
    
    if best_val > r_current:
        return candidates[best_idx], best_val
        
    return p_i, -1.0 # No improvement found

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    # 1. Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    grid_pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    initializations = [grid_pts]
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        rand_pts = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(rand_pts)
        
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Local Search Loop
        improved = True
        max_iter = 1000
        iter_count = 0
        
        while improved and iter_count < max_iter:
            if time.time() > deadline - 5:
                break
                
            improved = False
            iter_count += 1
            
            # Identify the "bottleneck" sphere(s)
            # The sphere(s) that are at the minimum distance to a neighbor or wall.
            
            # Calculate wall margins
            wall_margins = CUBE_HALF - np.abs(current_pts).max(axis=1)
            min_wall_margin = wall_margins.min()
            
            # Calculate pairwise distances
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = dists.min()
            half_min_pair_dist = min_pair_dist / 2.0
            
            current_r = min(min_wall_margin, half_min_pair_dist)
            
            # Find indices of spheres involved in the minimum
            # 1. Wall bottlenecks
            wall_bottlenecks = np.where(wall_margins <= current_r + 1e-9)[0]
            
            # 2. Pair bottlenecks
            # Find pairs with distance <= 2 * current_r + 1e-9
            # We can get the indices from the dists matrix
            # To avoid O(N^2) scan every time, we can just pick a random sphere?
            # No, picking the bottleneck is crucial for best-improvement.
            
            # Get all pairs with min distance
            # dists is (N, N). We want i,j such that dists[i,j] <= 2*current_r + eps
            # And i < j.
            
            # This is a bit expensive to do from scratch every iteration.
            # But N=100, so 100x100 matrix is small.
            
            # Let's just pick one bottleneck sphere at random from the set of all spheres
            # that are "tight".
            # A sphere is tight if its min distance to any other sphere or wall is <= current_r + eps.
            
            # For each sphere i:
            # min_dist_i = min(wall_margin_i, min_{j!=i} dist(i,j)/2)
            # If min_dist_i <= current_r + eps, then i is a bottleneck.
            
            # Compute min_dist for all spheres
            # min_pair_dists_per_sphere = dists.min(axis=1) / 2.0
            # min_dist_all = np.minimum(wall_margins, min_pair_dists_per_sphere)
            
            min_pair_dists_per_sphere = dists.min(axis=1) / 2.0
            min_dist_all = np.minimum(wall_margins, min_pair_dists_per_sphere)
            
            # Bottlenecks are those with min_dist_all close to current_r
            # We pick one at random to avoid bias, or the one with the smallest min_dist_all?
            # Best-improvement usually implies trying to fix the worst one.
            # But if we fix one, the others might become the worst.
            # Let's pick a random bottleneck to explore different directions.
            
            bottleneck_indices = np.where(min_dist_all <= current_r + 1e-6)[0]
            
            if len(bottleneck_indices) == 0:
                # This shouldn't happen if current_r is correctly calculated,
                # but due to floating point, maybe.
                # Pick the one with the smallest min_dist_all
                i = np.argmin(min_dist_all)
            else:
                i = bottleneck_indices[rng.integers(len(bottleneck_indices))]
                
            # Try to improve sphere i
            new_pos, new_val = _local_search_sphere(i, current_pts, current_r, deadline)
            
            if new_val > current_r:
                # Accept the move
                current_pts[i] = new_pos
                current_r = new_val
                improved = True
                current_score = current_r # Approximation, will be verified later
                
        # After local search converges, calculate exact score
        final_score = _score(current_pts)
        
        if final_score > best_score:
            best_score = final_score
            best_points = current_pts.copy()
            
        # If we have time, try a "kick" to escape local minimum
        if time.time() < deadline - 5 and final_score > 0.1:
            # Randomly perturb a few spheres
            kick_pts = current_pts.copy()
            num_kicks = 3
            kick_indices = rng.choice(N, size=num_kicks, replace=False)
            for idx in kick_indices:
                kick_pts[idx] += rng.uniform(-0.01, 0.01, 3)
                # Ensure inside cube
                kick_pts[idx] = np.clip(kick_pts[idx], -0.49, 0.49)
            
            # Run a few iterations of local search on the kicked config
            kick_score = _score(kick_pts)
            if kick_score > 0:
                # Simple local search on kick_pts
                k_improved = True
                k_iter = 0
                k_r = kick_score
                while k_improved and k_iter < 50:
                    if time.time() > deadline - 2:
                        break
                    k_improved = False
                    k_iter += 1
                    
                    wall_margins = CUBE_HALF - np.abs(kick_pts).max(axis=1)
                    diff = kick_pts[:, np.newaxis, :] - kick_pts[np.newaxis, :, :]
                    dists = np.sqrt((diff ** 2).sum(axis=2))
                    np.fill_diagonal(dists, np.inf)
                    min_pair_dists_per_sphere = dists.min(axis=1) / 2.0
                    min_dist_all = np.minimum(wall_margins, min_pair_dists_per_sphere)
                    k_r = min_dist_all.min()
                    
                    bottleneck_indices = np.where(min_dist_all <= k_r + 1e-6)[0]
                    if len(bottleneck_indices) > 0:
                        i = bottleneck_indices[rng.integers(len(bottleneck_indices))]
                        new_pos, new_val = _local_search_sphere(i, kick_pts, k_r, deadline)
                        if new_val > k_r:
                            kick_pts[i] = new_pos
                            k_r = new_val
                            k_improved = True
                
                final_kick_score = _score(kick_pts)
                if final_kick_score > best_score:
                    best_score = final_kick_score
                    best_points = kick_pts.copy()

    if best_points is None:
        best_points = grid_pts
        
    return best_points

# EVOLVE-BLOCK-END
