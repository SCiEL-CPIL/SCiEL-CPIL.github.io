# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 625  # Safety margin before 630s limit
    
    n = 33
    # Precompute C(29, 3) = 3654
    WEIGHT = 3654.0
    
    # Helper: Vectorized convexity check for a set of 4-tuples
    # points: (N, 2) array
    # indices: (M, 4) array of indices into points
    # Returns: boolean array of shape (M,) where True means convex
    def is_convex_batch(points, indices):
        # indices shape (M, 4)
        p0 = points[indices[:, 0]] # (M, 2)
        p1 = points[indices[:, 1]]
        p2 = points[indices[:, 2]]
        p3 = points[indices[:, 3]]
        
        # Calculate double areas of the 4 triangles
        # Area2(a, b, c) = (b-a) x (c-a)
        # We need absolute values for the sum check, but signs matter for orientation?
        # The condition "largest area == sum of others" works with absolute areas.
        
        # Triangle 012
        v1 = p1 - p0
        v2 = p2 - p0
        a012 = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Triangle 013
        v3 = p3 - p0
        a013 = np.abs(v1[:, 0] * v3[:, 1] - v1[:, 1] * v3[:, 0])
        
        # Triangle 023
        # (p2-p0) x (p3-p0)
        a023 = np.abs(v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])
        
        # Triangle 123
        # (p2-p1) x (p3-p1)
        v21 = p2 - p1
        v31 = p3 - p1
        a123 = np.abs(v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0])
        
        areas = np.stack([a012, a013, a023, a123], axis=1) # (M, 4)
        
        # For a non-convex quad (one point inside), the largest area equals the sum of the other three.
        # For a convex quad, the sum of the two "diagonal" triangles equals the sum of the other two?
        # Actually, for convex, Area(012) + Area(032) != Area(013) + Area(123) generally?
        # Let's stick to the "one inside" check.
        # If p0 is inside 123: a123 = a012 + a013 + a023
        # If p1 is inside 023: a023 = a012 + a013 + a123
        # If p2 is inside 013: a013 = a012 + a023 + a123
        # If p3 is inside 012: a012 = a013 + a023 + a123
        
        # We can check if any of these conditions hold (within tolerance).
        # Since floating point errors exist, we use a tolerance.
        # However, if points are in general position, the areas won't be exactly equal.
        # We check if the difference is small.
        
        tol = 1e-8
        # Check p0 inside
        cond0 = np.abs(a123 - (a012 + a013 + a023)) < tol
        # Check p1 inside
        cond1 = np.abs(a023 - (a012 + a013 + a123)) < tol
        # Check p2 inside
        cond2 = np.abs(a013 - (a012 + a023 + a123)) < tol
        # Check p3 inside
        cond3 = np.abs(a012 - (a013 + a023 + a123)) < tol
        
        is_non_convex = cond0 | cond1 | cond2 | cond3
        return ~is_non_convex

    # Generate all 4-combinations of indices
    # We need this for the delta calculation.
    # Instead of generating all C(33,4), we generate the C(32,3) subsets for a specific moved point.
    # But for the initial count, we need all C(33,4).
    
    # Precompute all 4-combinations for the full set?
    # C(33, 4) = 40920. This is small enough to store.
    all_combinations = np.array(list(zip(*np.meshgrid(
        np.arange(33), np.arange(33), np.arange(33), np.arange(33), indexing='ij'
    )))).reshape(-1, 4)
    # Filter valid combinations i < j < k < l
    mask = (all_combinations[:, 0] < all_combinations[:, 1]) & \
           (all_combinations[:, 1] < all_combinations[:, 2]) & \
           (all_combinations[:, 2] < all_combinations[:, 3])
    all_combinations = all_combinations[mask]
    
    # Function to count convex 7-gons given points
    def count_convex_7goms(points):
        convex_flags = is_convex_batch(points, all_combinations)
        return np.sum(convex_flags) * WEIGHT

    # Function to get delta for moving point idx
    def get_delta(points, idx, new_p):
        # Find all combinations containing idx
        # We can precompute the mask for each idx?
        # Or just filter. Filtering 40920 items 33 times is fast.
        # Better: Precompute a list of indices for each point.
        # Let's just filter here for simplicity, it's fast enough with NumPy.
        
        # Create a mask for combinations where idx is present
        # all_combinations is (40920, 4)
        # We want rows where any column is idx.
        # But we must ensure we don't double count? No, each 4-tuple is unique.
        # If idx is in the tuple, it's in exactly one position.
        
        # Efficient way:
        # Precompute for each point i, the list of combination indices that contain i.
        # This saves filtering time.
        
        # Let's assume we have precomputed `comb_indices_for_point`
        # Since we are inside the function, we can't easily pass closures if we want to keep it clean.
        # We'll compute the mask.
        
        # Actually, creating a boolean mask of size 40920 is very fast.
        # But doing it 1000s of times might add up.
        # Let's precompute the "indices of combinations containing point i" for all i.
        
        pass # Placeholder, will implement in main block

    # Precompute combination indices for each point
    # comb_indices[i] is an array of indices into all_combinations where point i appears
    comb_indices = []
    for i in range(n):
        mask = np.any(all_combinations == i, axis=1)
        comb_indices.append(np.where(mask)[0])
        
    def get_delta_vectorized(points, idx, new_p):
        # Get the specific combinations for this point
        comb_idx = comb_indices[idx]
        if len(comb_idx) == 0:
            return 0.0
            
        # Extract the 4-tuples for these combinations
        # We need to replace idx with new_p in the tuple
        # all_combinations[comb_idx] has shape (M, 4)
        # We need to identify which column is idx and replace it.
        
        tuples = all_combinations[comb_idx].copy()
        
        # Find where idx is in each tuple
        # tuples is (M, 4). We want to replace tuples[m, col] with new_p if tuples[m, col] == idx.
        # Since idx is unique in the tuple, we can just find the column.
        
        # Create a mask for the position of idx
        pos_mask = (tuples == idx) # (M, 4)
        
        # Create the new points array for these tuples
        # We can't just replace in tuples directly because tuples are indices.
        # We need to evaluate convexity for the OLD points and NEW points.
        
        # OLD convexity:
        # We already know the old convexity? No, we don't store it.
        # We can compute it.
        # But we can just compute the NEW convexity and compare to a stored global state?
        # Storing global state (convexity of all 40920 quads) is efficient.
        
        # Let's maintain `current_convex_flags` array of shape (40920,)
        # When we move a point, we update the flags for the affected combinations.
        
        return 0.0 # Placeholder

    # Revised Strategy:
    # 1. Initialize points.
    # 2. Compute initial convex flags for all 40920 combinations.
    # 3. In SA loop:
    #    a. Pick random point idx.
    #    b. Generate new position.
    #    c. Check validity (distance/collinearity) - vectorized.
    #    d. Compute new convexity for the ~4960 combinations involving idx.
    #    e. Compare with current flags.
    #    f. Delta = (New_Convex_Count - Old_Convex_Count) * WEIGHT.
    #    g. Accept/Reject.
    #    h. If accepted, update the global convex flags for those combinations.

    # Initial Construction: 6x6 Grid minus 3
    random.seed(42)
    np.random.seed(42)
    
    # Create 6x6 grid
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 6))
    grid_pts = np.column_stack((xs.ravel(), ys.ravel()))
    
    # Remove 3 points. Which ones?
    # Removing corners might help? Or center?
    # Let's remove 3 random points to break symmetry.
    remove_idx = np.random.choice(36, 3, replace=False)
    keep_mask = np.ones(36, dtype=bool)
    keep_mask[remove_idx] = False
    pts = grid_pts[keep_mask].copy()
    
    # Add small jitter to avoid exact collinearities/distances
    pts += np.random.uniform(-0.01, 0.01, size=pts.shape)
    pts = np.clip(pts, 0.0, 1.0)

    # Validate initial points
    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        # Vectorized
        # dist_sq = (points[:, None, :] - points[None, :, :])**2
        # This creates a 33x33x2 array.
        diffs = points[:, None, :] - points[None, :, :]
        dist_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        if np.min(dist_sq) < 1e-12:
            return False
        # Collinearity
        # Check all triples. 33C3 = 5456.
        # We can vectorize this too?
        # For each triple (i, j, k), compute area.
        # This is harder to fully vectorize without loops or large arrays.
        # 5456 is small, loop is fine.
        for i in range(33):
            for j in range(i+1, 33):
                for k in range(j+1, 33):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    if not is_valid(pts):
        # Fallback
        pts = np.random.rand(33, 2) * 0.98 + 0.01
        if not is_valid(pts):
             # Extreme fallback
             return np.random.rand(33, 2)

    # Compute initial convex flags
    current_convex_flags = is_convex_batch(pts, all_combinations)
    current_count = np.sum(current_convex_flags) * WEIGHT
    best_points = pts.copy()
    best_count = current_count
    best_flags = current_convex_flags.copy()

    # SA Parameters
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    step_size = 0.05
    time_limit = deadline - 5

    # Precompute validity checks for speed?
    # Distance check:
    def check_validity_fast(points, idx, new_p):
        # Check distance to all others
        diffs = points - new_p
        dist_sq = np.sum(diffs**2, axis=1)
        dist_sq[idx] = np.inf
        if np.min(dist_sq) < 1e-12:
            return False
            
        # Check collinearity with all pairs
        # We need to check if new_p is collinear with any pair (i, j) from points.
        # This is C(32, 2) = 496 checks.
        # Vectorized:
        # For each pair (i, j), compute area of triangle (new_p, p_i, p_j).
        # We can create a matrix of pairs.
        i_idx, j_idx = np.triu_indices(33, k=1)
        # Exclude pairs involving idx
        mask = (i_idx != idx) & (j_idx != idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        p_i = points[i_idx]
        p_j = points[j_idx]
        
        # Area = 0.5 * | (p_i - new_p) x (p_j - new_p) |
        v1 = p_i - new_p
        v2 = p_j - new_p
        areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        if np.any(areas < 1e-9):
            return False
            
        return True

    # Main Loop
    while time.time() < time_limit:
        idx = random.randint(0, 32)
        
        # Generate new point
        old_p = best_points[idx]
        new_p = old_p + np.random.normal(0, step_size, 2)
        new_p = np.clip(new_p, 0.0, 1.0)
        
        # Quick validity check
        if not check_validity_fast(best_points, idx, new_p):
            continue
            
        # Calculate Delta
        comb_idx = comb_indices[idx]
        if len(comb_idx) == 0:
            continue
            
        # Get the tuples
        tuples = all_combinations[comb_idx]
        
        # We need to evaluate convexity for these tuples with the NEW point.
        # The tuples contain indices. We need to substitute the coordinate of idx with new_p.
        # Instead of modifying the global points array, we can create a temporary points array?
        # No, that's slow.
        # We can modify the `is_convex_batch` to accept a "replacement" point?
        # Or, simpler: Create a copy of points, modify idx, call is_convex_batch on the subset.
        # Copying 33x2 array is fast.
        
        temp_pts = best_points.copy()
        temp_pts[idx] = new_p
        
        new_flags = is_convex_batch(temp_pts, tuples)
        
        # Old flags for these combinations
        old_flags = best_flags[comb_idx]
        
        # Delta in number of convex quads
        delta_quads = np.sum(new_flags) - np.sum(old_flags)
        delta_count = delta_quads * WEIGHT
        
        # Acceptance
        if delta_count < 0 or random.random() < math.exp(-delta_count / T):
            # Accept
            best_points[idx] = new_p
            best_flags[comb_idx] = new_flags
            current_count += delta_count
            if current_count < best_count:
                best_count = current_count
                best_points_saved = best_points.copy()
                best_flags_saved = best_flags.copy()
                
            T = max(T_min, T * cooling)
        else:
            T = max(T_min, T * cooling)
            
        # Adjust step size
        if random.random() < 0.01:
            step_size = max(0.01, step_size * 0.9)

    # Final Validation
    if not is_valid(best_points):
        return best_points_saved if 'best_points_saved' in locals() else best_points

    return best_points

# EVOLVE-BLOCK-END
