# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Attempts to find a large set of integer points in 11 dimensions such that
    the minimum squared distance between any pair is >= the maximum squared norm.
    
    Strategy:
    1. Iterate through candidate points in increasing order of norm squared.
    2. For a given maximum allowed norm squared R, greedily select points 
       from the set of all integer vectors with norm squared <= R.
    3. A point is accepted if its squared distance to all previously accepted 
       points is >= R.
    4. We try different values of R to see which yields the most points.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 550  # Leave 50s buffer for safety
    
    best_points = []
    best_k = 0
    
    # We will search for configurations with max_norm_sq (R) up to a certain limit.
    # The current seed uses R=2. 
    # Let's try R=2, 3, 4, 5... 
    # Note: The number of integer points with norm squared <= R grows with R.
    # However, the constraint ||p_i - p_j||^2 >= R becomes stricter as R increases.
    
    # Pre-compute candidate points for various R values.
    # To be efficient, we generate candidates for the largest R we want to try 
    # and filter down, or generate per R.
    # Given the small dimension and likely small R for high density, 
    # generating up to R=10 is feasible.
    
    max_R_to_try = 12
    
    # Generate all integer points with norm squared <= max_R_to_try
    # We can do this by iterating over all combinations of coordinates.
    # Since norm is small, coordinates are bounded by sqrt(max_R_to_try).
    bound = int(np.sqrt(max_R_to_try)) + 1
    
    # Generate all points in [-bound, bound]^d with norm^2 <= max_R_to_try
    # This is too many if we just iterate all d-dim ranges. 
    # Better: generate by weight.
    
    candidates_by_norm = {r: [] for r in range(1, max_R_to_try + 1)}
    
    # Efficient generation of integer points with small norm
    # We can iterate through all subsets of coordinates to be non-zero.
    # For d=11, subsets are manageable for small weights.
    # Max weight is bounded by max_R_to_try (if all 1s) or less.
    
    for weight in range(1, d + 1):
        # If weight > max_R_to_try, break (min norm for weight w is w)
        if weight > max_R_to_try:
            break
            
        # Iterate over all combinations of indices of size 'weight'
        for indices in itertools.combinations(range(d), weight):
            # For each combination, we need to find integer values x_i such that sum x_i^2 <= max_R_to_try
            # And x_i != 0.
            
            # Recursive helper to fill values
            def generate_values(idx_pos, current_vals, remaining_sum):
                if idx_pos == weight:
                    if remaining_sum == 0:
                        # Create the point
                        p = np.zeros(d, dtype=np.int64)
                        for k_idx, val in zip(indices, current_vals):
                            p[k_idx] = val
                        norm_sq = int(np.dot(p, p))
                        if 1 <= norm_sq <= max_R_to_try:
                            candidates_by_norm[norm_sq].append(p)
                    return
                
                # Try all possible non-zero values for this coordinate
                # Value v must satisfy v^2 <= remaining_sum
                # v can be positive or negative
                max_val = int(np.sqrt(remaining_sum))
                for v in range(1, max_val + 1):
                    # Try +v
                    generate_values(idx_pos + 1, current_vals + [v], remaining_sum - v*v)
                    # Try -v
                    generate_values(idx_pos + 1, current_vals + [-v], remaining_sum - v*v)
            
            generate_values(0, [], max_R_to_try)

    # Now we have candidates_by_norm. 
    # For a target R, the set of available points is all points with norm_sq <= R.
    # But the condition is: min_dist_sq >= R.
    # And all points must have norm_sq <= R (by definition of max_norm_sq being R? 
    # Actually, max_norm_sq is determined by the points. If we pick a point with norm > R, 
    # then max_norm_sq becomes that norm, and the distance constraint tightens.
    # So it's optimal to consider sets where max_norm_sq = R.
    # This implies all points have norm_sq <= R, and at least one has norm_sq = R?
    # Not necessarily, but if all are < R, we could potentially add more or it's a subset of a smaller R case.
    # So we iterate R from small to large.
    
    # Sort candidates by norm to allow incremental addition? 
    # Greedy approach:
    # For each R in [2, 3, ..., max_R_to_try]:
    #   Let S be the set of all points with norm_sq <= R.
    #   We want to find a maximal subset P of S such that for all p_i, p_j in P, ||p_i - p_j||^2 >= R.
    #   This is a maximum independent set problem in a graph where edges connect points 
    #   with distance^2 < R.
    #   Since we want to maximize count, and the graph is defined by distance,
    #   a greedy algorithm picking points in some order might work, but is not guaranteed optimal.
    #   However, given the symmetry and structure, a simple greedy might find a good solution.
    #   
    #   To make it deterministic and fast, we can shuffle or sort candidates.
    #   Sorting by norm (smallest first) might help pack tighter? Or largest first?
    #   Usually, for packing, taking points that are "far" from others is key.
    #   
    #   Let's try a greedy:
    #   1. Get all candidates with norm_sq <= R.
    #   2. Shuffle them (or sort by norm, then index) to break symmetry.
    #   3. Iterate through candidates.
    #   4. If candidate is at distance^2 >= R from all currently selected points, add it.
    #   5. If time is running out, stop.
    
    # Optimization: 
    # The number of candidates for small R is small.
    # R=2: 220 points.
    # R=3: Points with norm 2 or 3. Norm 3: (1,1,1) type. 
    # R=4: Points with norm 2,3,4. Norm 4: (2,0..) or (1,1,1,1).
    
    # Let's run the greedy for each R.
    
    # To avoid re-checking all pairs every time, we can maintain the selected points list.
    # Check cost: O(k * d) per new point.
    
    for R in range(2, max_R_to_try + 1):
        if time.time() > deadline:
            break
            
        # Collect all candidates with norm_sq <= R
        # Note: candidates_by_norm[r] contains points with EXACTLY norm r.
        # We need the union of 1..R.
        # But we should probably only consider points that are "good" candidates.
        # Actually, if a point has norm < R, it's in the candidate pool.
        
        pool = []
        for r in range(1, R + 1):
            pool.extend(candidates_by_norm[r])
        
        if not pool:
            continue
            
        # Deduplicate? The generation logic might produce duplicates? 
        # No, each point is generated uniquely by its non-zero indices and values.
        # But wait, the recursive generator iterates over indices combinations.
        # A point like (1, 0, 1, ...) is generated only once for the specific index set.
        # So no duplicates.
        
        # Sort pool by norm, then lexicographically for determinism
        # Using norm as primary key might help? 
        # Let's just use a fixed order to be deterministic.
        # pool.sort(key=lambda p: (int(np.dot(p,p)), tuple(p)))
        
        selected = []
        
        # To speed up distance checks, we can use numpy broadcasting if selected is small,
        # but for large k, loop is fine if k is not huge.
        # k is likely < 1000.
        
        for p in pool:
            # Check if p is far enough from all selected points
            valid = True
            for s in selected:
                diff = p - s
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < R:
                    valid = False
                    break
            
            if valid:
                selected.append(p)
                
        k = len(selected)
        if k > best_k:
            best_k = k
            best_points = selected
            
    # Convert best_points to numpy array
    if not best_points:
        # Fallback to the known good seed solution (weight 2 vectors)
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        points.append(p)
        best_points = points
        best_k = len(best_points)

    result = np.array(best_points, dtype=np.int64)
    
    # Final verification and rounding as per contract
    # The contract says: "Entries are rounded with np.around and cast to int64"
    # Since we are using integers, this is a no-op, but we do it to be safe.
    result = np.around(result).astype(np.int64)
    
    return result

# EVOLVE-BLOCK-END
