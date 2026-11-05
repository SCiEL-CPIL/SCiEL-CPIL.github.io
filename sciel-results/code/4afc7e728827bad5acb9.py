"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    deadline = time.time() + 285  # Safety margin before 330s kill
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Precompute squared distances as int32 matrix
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Get unique squared distances and map to small integers
    # For N=64, max dist2 = 63^2 + 63^2 = 7938
    # Number of possible values is at most ~8000
    max_dist2 = int(dist2.max())
    
    # Create a lookup: for each point i, a boolean mask of which dist2 values it has to other points
    # We'll use a sparse representation: for each point in S, store a set of dist2 values
    
    # For fast validation, precompute for each point P, the sorted list of (dist2, point_index) pairs
    # But that's 4096 * 4096 = 16M entries, too much memory for sorted lists
    
    # Alternative: use the dist2 matrix directly with vectorized checks
    
    # State representation:
    # S: list of point indices in current set
    # dist_mask[i]: np.array of bool, length max_dist2+1, True if point i has that dist2 to some other point in S
    # Actually, let's use sets for efficiency since |S| is small (~10-20)
    
    # For vectorized validation of a candidate P against current set S:
    # 1. Get all dist2[P, S] values
    # 2. Check they're all unique (no isosceles with P as apex)
    # 3. For each Q in S, check dist2[P,Q] is not in Q's existing dist set
    
    # Let's use a different approach: maintain for each point in S, a set of dist2 values
    # And use vectorized numpy for the uniqueness check
    
    def validate_add(P, S_indices, dist_sets):
        """
        Check if P can be added.
        S_indices: np.array of indices in S
        dist_sets: list of sets, dist_sets[i] = set of dist2 values from S_indices[i] to other points in S
        Returns: True if valid
        """
        if len(S_indices) == 0:
            return True
        
        # Get distances from P to all points in S
        d_vals = dist2[P, S_indices]  # shape (|S|,)
        
        # Check uniqueness: no two points in S at same distance from P
        if len(np.unique(d_vals)) != len(d_vals):
            return False
        
        # Check each d_val is not in the corresponding point's dist set
        for i, idx_Q in enumerate(S_indices):
            if int(d_vals[i]) in dist_sets[i]:
                return False
        
        return True
    
    def add_to_set(P, S_indices, dist_sets):
        """Add P to the set, updating dist_sets."""
        S_indices = np.append(S_indices, P)
        new_dist_set = set()
        for i, idx_Q in enumerate(S_indices[:-1]):
            d = int(dist2[P, idx_Q])
            new_dist_set.add(d)
            dist_sets[i].add(d)
        dist_sets.append(new_dist_set)
        return S_indices
    
    def remove_from_set(P, S_indices, dist_sets):
        """Remove P from the set, updating dist_sets."""
        # Find position of P in S_indices
        pos = None
        for i, idx in enumerate(S_indices):
            if idx == P:
                pos = i
                break
        if pos is None:
            return S_indices
        
        # Remove distances from other points
        for i, idx_Q in enumerate(S_indices):
            if i == pos:
                continue
            d = int(dist2[P, idx_Q])
            dist_sets[i].discard(d)
        
        # Remove P's dist set
        dist_sets.pop(pos)
        
        # Remove P from S_indices
        S_indices = np.delete(S_indices, pos)
        return S_indices
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction with multiple orderings
    best_S = np.array([], dtype=np.int32)
    best_dist_sets = []
    
    orderings = []
    
    # Random order
    order = np.arange(num_points)
    rng.shuffle(order)
    orderings.append(order)
    
    # Sorted by x
    orderings.append(np.array(sorted(range(num_points), key=lambda i: points[i][0])))
    
    # Sorted by y
    orderings.append(np.array(sorted(range(num_points), key=lambda i: points[i][1])))
    
    # Sorted by x+y
    orderings.append(np.array(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1])))
    
    # Sorted by distance from center
    cx, cy = N/2.0, N/2.0
    orderings.append(np.array(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)))
    
    # Spiral-like ordering
    def spiral_order():
        order = []
        min_x, max_x = 0, N-1
        min_y, max_y = 0, N-1
        while min_x <= max_x and min_y <= max_y:
            for x in range(min_x, max_x+1):
                order.append(min_y * N + x)
            min_y += 1
            for y in range(min_y, max_y+1):
                order.append(y * N + max_x)
            max_x -= 1
            if min_y <= max_y:
                for x in range(max_x, min_x-1, -1):
                    order.append(max_y * N + x)
                max_y -= 1
            if min_x <= max_x:
                for y in range(max_y, min_y-1, -1):
                    order.append(y * N + min_x)
                min_x += 1
        return np.array(order)
    
    orderings.append(spiral_order())
    
    for order in orderings:
        S = np.array([], dtype=np.int32)
        dist_sets = []
        in_S = np.zeros(num_points, dtype=bool)
        
        for P in order:
            if in_S[P]:
                continue
            if validate_add(P, S, dist_sets):
                S = add_to_set(P, S, dist_sets)
                in_S[P] = True
        
        if len(S) > len(best_S):
            best_S = S.copy()
            best_dist_sets = [ds.copy() for ds in dist_sets]
    
    # Phase 2: Local search with vectorized validation
    S = best_S.copy()
    dist_sets = [ds.copy() for ds in best_dist_sets]
    in_S = np.zeros(num_points, dtype=bool)
    for idx in S:
        in_S[idx] = True
    
    complement = np.where(~in_S)[0]
    best_size = len(S)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5:
        if len(complement) == 0:
            break
        
        # Try to add a point
        # Sample batch of candidates and validate
        batch_size = min(50, len(complement))
        candidates = rng.sample(list(complement), batch_size)
        
        added = False
        for P in candidates:
            if validate_add(P, S, dist_sets):
                S = add_to_set(P, S, dist_sets)
                in_S[P] = True
                complement = np.delete(complement, np.where(complement == P)[0][0])
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_dist_sets = [ds.copy() for ds in dist_sets]
                    no_improve = 0
                break
        
        if not added:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Try swaps: remove a point, then try to add two new points
            if len(S) > 2:
                # Remove a random point
                rem_pos = rng.randrange(len(S))
                rem_P = S[rem_pos]
                S = remove_from_set(rem_P, S, dist_sets)
                in_S[rem_P] = False
                complement = np.append(complement, rem_P)
                
                # Try to add two new points
                added_count = 0
                for _ in range(2):
                    if len(complement) == 0:
                        break
                    batch_size = min(30, len(complement))
                    candidates = rng.sample(list(complement), batch_size)
                    for P in candidates:
                        if validate_add(P, S, dist_sets):
                            S = add_to_set(P, S, dist_sets)
                            in_S[P] = True
                            comp_idx = np.where(complement == P)[0]
                            if len(comp_idx) > 0:
                                complement = np.delete(complement, comp_idx[0])
                            added_count += 1
                            break
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_dist_sets = [ds.copy() for ds in dist_sets]
                    no_improve = 0
                else:
                    no_improve = 0  # Reset after perturbation
            else:
                # Reset to best
                S = best_S.copy()
                dist_sets = [ds.copy() for ds in best_dist_sets]
                in_S = np.zeros(num_points, dtype=bool)
                for idx in S:
                    in_S[idx] = True
                complement = np.where(~in_S)[0]
                no_improve = 0
    
    # Return best solution
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
