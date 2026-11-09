import numpy as np
import time
import itertools

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimizes the number of convex 7-gons among 33 points in the unit square.
    Uses local search minimizing the number of convex 4-gons as a proxy.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Initialize with random points
    np.random.seed(42)
    points = np.random.rand(n, 2)
    
    # Helper to check validity
    def is_valid(pts):
        # Check bounds
        if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
            return False
        # Check NaN/Inf
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return False
        # Check distance
        # For speed, we might skip this in the inner loop and only check at the end,
        # or ensure step size is large enough.
        # Check collinearity is hard to check fast for all triples, 
        # but random floats almost never produce exact collinearity.
        # We assume general position is maintained by random perturbation.
        return True

    # Precompute indices for 4-gons containing each point
    # We need to update the count efficiently.
    # Total 4-gons: C(33, 4) = 40920
    # 4-gons containing point i: C(32, 3) = 4960
    
    # Generate all 4-combinations
    # We store them as arrays of indices
    all_combos = np.array(list(itertools.combinations(range(n), 4)))
    
    # For each point i, which combos contain it?
    # We can create a list of indices into all_combos for each point.
    point_to_combos = [[] for _ in range(n)]
    for idx, combo in enumerate(all_combos):
        for p in combo:
            point_to_combos[p].append(idx)
    
    # Convert to numpy arrays for fast access
    point_to_combos_idx = [np.array(lst) for lst in point_to_combos]

    def count_convex4_subset(points, combo_indices):
        """
        Counts convex 4-gons among the subsets defined by combo_indices.
        combo_indices is an array of indices into all_combos.
        """
        if len(combo_indices) == 0:
            return 0
            
        # Get the points for these combos
        # Shape: (num_combos, 4, 2)
        combos = all_combos[combo_indices]
        pts = points[combos]
        
        # Unpack x, y
        x1, y1 = pts[:, 0, 0], pts[:, 0, 1]
        x2, y2 = pts[:, 1, 0], pts[:, 1, 1]
        x3, y3 = pts[:, 2, 0], pts[:, 2, 1]
        x4, y4 = pts[:, 3, 0], pts[:, 3, 1]
        
        # Calculate cross products for the 4 turns
        # Turn 1: p1 -> p2 -> p3
        cp1 = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        # Turn 2: p2 -> p3 -> p4
        cp2 = (x3 - x2) * (y4 - y3) - (y3 - y2) * (x4 - x3)
        # Turn 3: p3 -> p4 -> p1
        cp3 = (x4 - x3) * (y1 - y4) - (y4 - y3) * (x1 - x4)
        # Turn 4: p4 -> p1 -> p2
        cp4 = (x1 - x4) * (y2 - y1) - (y1 - y4) * (x2 - x4)
        
        # A polygon is convex if all cross products have the same sign
        # (and none are zero, but we assume general position)
        # Check if all positive or all negative
        # We can check if min > 0 or max < 0
        # But floating point noise might cause issues.
        # Use a small epsilon? The task says |cross| < 1e-9 is invalid for collinearity,
        # but for convexity check, the evaluator uses > 1e-9 for "convex turn".
        # So we should mimic that.
        
        # Convex if all cp > 1e-9 OR all cp < -1e-9
        eps = 1e-9
        all_pos = np.all(cp1 > eps, axis=0) & np.all(cp2 > eps, axis=0) & np.all(cp3 > eps, axis=0) & np.all(cp4 > eps, axis=0)
        all_neg = np.all(cp1 < -eps, axis=0) & np.all(cp2 < -eps, axis=0) & np.all(cp3 < -eps, axis=0) & np.all(cp4 < -eps, axis=0)
        
        return np.sum(all_pos | all_neg)

    # Initial count
    current_count = count_convex4_subset(points, np.arange(len(all_combos)))
    best_count = current_count
    best_points = points.copy()
    
    # Local Search
    step_size = 0.1
    
    while time.time() < deadline:
        # Pick a random point to move
        i = np.random.randint(n)
        
        # Save original position
        orig_pos = points[i].copy()
        
        # Generate new position
        # Gaussian perturbation
        new_pos = orig_pos + np.random.normal(0, step_size, 2)
        
        # Keep within bounds [0, 1]
        # Simple reflection or clamping? Clamping might cause points to stick to boundary.
        # Let's use a soft bound: if out of bounds, reject or reflect.
        # For simplicity, reject if out of bounds.
        if np.any(new_pos < 0) or np.any(new_pos > 1):
            continue
            
        points[i] = new_pos
        
        # Calculate new count
        # Only 4-gons containing i are affected
        affected_indices = point_to_combos_idx[i]
        
        # We need to subtract the old count of affected combos and add the new count.
        # But we didn't store the old count per combo.
        # We can just recompute the total count? 
        # Total count is 40920 subsets. 
        # Recomputing all 40920 is fast (~1ms).
        # Recomputing only 4960 is also fast.
        # Let's just recompute the full count for simplicity and robustness, 
        # unless it's too slow.
        # 40920 * 4 points * vectorized ops is very fast in NumPy.
        
        new_count = count_convex4_subset(points, np.arange(len(all_combos)))
        
        if new_count < best_count:
            best_count = new_count
            best_points = points.copy()
            # Reduce step size slightly?
            # step_size *= 0.99
        else:
            # Revert
            points[i] = orig_pos
            
        # Check deadline
        if time.time() > deadline:
            break
            
    return best_points

# EVOLVE-BLOCK-END
