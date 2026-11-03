import numpy as np
import time

N = 300

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing for 300 circles
    # Area of unit circle = pi. Area per circle ~ pi / 300.
    # Radius ~ sqrt(pi/300 / pi) = 1/sqrt(300) ~ 0.057
    # Diameter ~ 0.114
    # Spacing in hex grid ~ diameter / sqrt(3) ~ 0.065
    # Let's try a range of spacings to find the best initial grid count
    
    best_points = None
    best_score = 0.0
    
    # Generate initial points on hex grid
    # We want to fill the disk.
    # Let's try a few grid densities
    for spacing_mult in [0.9, 1.0, 1.1]:
        spacing = 0.06 * spacing_mult
        # Generate grid
        # Hex grid coordinates
        # x = (i + j/2) * spacing
        # y = j * spacing * sqrt(3)/2
        max_k = int(1.0 / spacing) + 2
        points_list = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points_list.append((x, y))
        
        points = np.array(points_list)
        if len(points) == 0:
            continue
            
        # If we have more than N points, we need to select N.
        # If we have fewer, we need to add more (randomly?) or use a smaller spacing.
        # For now, if len(points) >= N, we can select the N points that are "most central" or just random.
        # A better selection: maximize the minimum distance of the selected subset.
        # This is a subset selection problem. 
        # Simple heuristic: keep points with largest distance to other points?
        # Or just take the first N. The grid is uniform, so any N should be similar.
        # Let's take the N points closest to the origin to ensure they fit well?
        # Or random selection from the grid points.
        
        if len(points) >= N:
            # Select N points. To maximize potential, we might want to spread them out.
            # But for initialization, random is fine.
            indices = np.random.choice(len(points), N, replace=False)
            p = points[indices]
        else:
            # Need to add points. Fill with random points in the disk.
            # This initialization is weak. We should adjust spacing to get ~N points.
            # Let's skip this initialization if count is too low.
            continue
            
        # Calculate initial score
        # Vectorized distance calculation
        # p is (N, 2)
        # dists = np.linalg.norm(p[:, np.newaxis] - p[np.newaxis, :], axis=-1)
        # This is O(N^2) memory. 300x300 is small.
        
        def calc_score(pts):
            # Check boundary
            norms = np.linalg.norm(pts, axis=1)
            min_bound = 1.0 - np.min(norms)
            if min_bound <= 0:
                return 0.0
                
            # Check pairwise
            # Use broadcasting
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            r = min(min_dist / 2.0, min_bound)
            return r

        score = calc_score(p)
        if score > best_score:
            best_score = score
            best_points = p.copy()

    if best_points is None:
        # Fallback: Random
        best_points = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(best_points, axis=1)
        best_points = best_points / norms[:, np.newaxis] * 0.99 # Push inside
        
    # 2. Simulated Annealing
    points = best_points
    current_score = best_score
    best_score_sa = current_score
    best_points_sa = points.copy()
    
    T = 0.001 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    max_steps = 1000000
    
    # Precompute some helpers?
    # We need to update score efficiently.
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Identify active points
        # Compute current distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        
        if r > best_score_sa:
            best_score_sa = r
            best_points_sa = points.copy()
            
        if r <= 0:
            # Invalid, revert or skip
            # In SA, we usually don't keep invalid states as "best"
            # But we need to handle the transition.
            # If r is 0, the move was bad.
            pass

        # Determine which points to move
        # Find indices of the closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        # Find index of the point closest to boundary
        k = np.argmin(bound_dists)
        
        # Active set: i, j, k
        active_idx = np.unique([i, j, k])
        
        # Perturb active points
        # Random direction
        noise = np.random.normal(0, T, size=(len(active_idx), 2))
        
        # Apply perturbation
        new_points = points.copy()
        new_points[active_idx] += noise
        
        # Calculate new score for new_points
        # Only need to check distances involving active points?
        # No, moving one point affects distances to ALL others.
        # So we must recalculate the global min.
        # Optimization: 
        # 1. Update rows/cols of dist matrix for active points.
        # 2. Update norms for active points.
        # 3. Find new global min.
        
        # Let's just recalculate for simplicity, N=300 is small.
        new_diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        new_dists = np.sqrt(np.sum(new_diff**2, axis=-1))
        np.fill_diagonal(new_dists, np.inf)
        
        new_norms = np.linalg.norm(new_points, axis=1)
        new_bound_dists = 1.0 - new_norms
        
        new_min_pair = np.min(new_dists)
        new_min_bound = np.min(new_bound_dists)
        new_r = min(new_min_pair / 2.0, new_min_bound)
        
        # Acceptance criterion
        if new_r > r:
            points = new_points
            current_score = new_r
        else:
            # Accept with prob exp((new_r - r) / T)
            if np.random.rand() < np.exp((new_r - r) / T):
                points = new_points
                current_score = new_r
                
        T *= alpha
        if T < T_min:
            break

    # 3. Polish
    # Run a few steps of very small perturbation or gradient ascent
    # Just do a few more SA steps with very low T
    T = 1e-7
    for step in range(1000):
        if time.time() - start_time > time_limit - 1:
            break
        # Same logic as above, but T is tiny
        # ... (reuse logic or skip for brevity if time is tight)
        
    # Ensure validity of best_points_sa
    # Check if best_points_sa is valid
    # If not, use the last valid state from SA?
    # The "best" score is only updated if r > best_score_sa.
    # If r was 0 (invalid), it wouldn't update best_score_sa unless best_score_sa was also 0 or negative.
    # Initial score was > 0. So best_points_sa should be valid.
    
    return best_points_sa
