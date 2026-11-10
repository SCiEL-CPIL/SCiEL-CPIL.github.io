import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for grid points
    grid_sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute the 4x4 matrix rows for all pairs (i, j) where i is the "center" p and j is a candidate q?
    # No, the matrix depends on p (the point being added) and the 4 existing points.
    # M rows: (q_i - p, |q_i|^2 - |p|^2)
    # This is affine in p and q.
    
    def get_validity_matrix(p_idx, q_indices):
        """
        Returns the (m, 4) matrix W where row i corresponds to q_indices[i].
        W[i] = [q_i - p, |q_i|^2 - |p|^2]
        """
        p = grid[p_idx]
        p_sq = grid_sq_norms[p_idx]
        qs = grid[q_indices]
        qs_sq = grid_sq_norms[q_indices]
        
        w = np.empty((len(q_indices), 4), dtype=np.float64)
        w[:, 0:3] = qs - p
        w[:, 3] = qs_sq - p_sq
        return w

    def check_all_dets(w):
        """
        Checks if any 4 rows of w have |det| < 0.5.
        Returns True if valid (no violation), False otherwise.
        w is (m, 4).
        """
        m = w.shape[0]
        if m < 4:
            return True
        
        # For efficiency, we check combinations.
        # If m is small, loop. If m is large, this is the bottleneck.
        # We will use a batched determinant check.
        
        # To speed up, we can skip if m is very large and we expect violations?
        # No, we need exactness.
        
        # Optimization: If we find a violation early, return False.
        
        # Batch processing
        batch_size = 2000
        # We need to iterate over combinations of 4 indices from 0..m-1
        # Generating all combinations for m=100 is C(100,4) = ~4M. Too many.
        # We need a smarter check or a faster det.
        
        # Alternative: Random sampling of combinations if m is large?
        # Risky for correctness.
        
        # Let's stick to exact but optimize the inner loop.
        # If m > 60, C(m,4) is > 400k. 
        # np.linalg.det on 4x4 is fast, but Python loop over combinations is slow.
        
        # Let's try to vectorize the combination generation if possible, or use a C-extension? No.
        # Let's use the fact that if ANY 4 points are co-spherical, the rank of the lifted matrix is < 5.
        # But we only care about subsets of size 4.
        
        # Strategy:
        # 1. If m <= 20: Loop all combinations.
        # 2. If m > 20: 
        #    We can use a heuristic: If the set is "dense", violations are likely.
        #    But we need to be exact.
        
        # Let's try a different approach for large m:
        # We can check the determinant of the full (m, 5) lifted matrix?
        # If rank < 5, all points are on a sphere.
        # But we only need *some* 4 points to define a sphere containing p.
        
        # Actually, the condition "no 5 points lie on a common sphere" is equivalent to:
        # For any set of 5 points, the rank of their lifted matrix is 5.
        # The current check: For new point p and existing set S, check if there exists Q subset of S, |Q|=4, such that rank(lifted(Q U {p})) < 5.
        
        # This is hard to vectorize over Q.
        
        # Let's optimize the combination loop using NumPy's advanced indexing in batches.
        
        def batch_check_dets(w, batch_size=5000):
            m = w.shape[0]
            # Generate indices for combinations
            # We can't generate all at once if m is large.
            # We will iterate in chunks of combinations.
            
            # itertools.combinations returns tuples.
            # We can batch them into a numpy array.
            
            # For m=50, C(50,4) = 230,300.
            # 230k determinants. 
            # np.linalg.det on (230k, 4, 4) might be memory heavy (230k * 16 floats = 3.6 MB). That's fine.
            # But creating the (230k, 4, 4) array is the cost.
            
            # Let's do it in smaller batches to keep memory low and allow early exit.
            
            combos = itertools.combinations(range(m), 4)
            batch = []
            for c in combos:
                batch.append(c)
                if len(batch) >= batch_size:
                    idx = np.array(batch, dtype=np.int32)
                    # idx is (B, 4)
                    # We need w[idx[i,0]], w[idx[i,1]], etc.
                    # This creates a (B, 4, 4) array.
                    # w is (m, 4).
                    # w[idx] where idx is (B, 4) gives (B, 4, 4).
                    mats = w[idx]
                    dets = np.linalg.det(mats)
                    if np.any(np.abs(dets) < 0.5):
                        return False
                    batch = []
            
            if batch:
                idx = np.array(batch, dtype=np.int32)
                mats = w[idx]
                dets = np.linalg.det(mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
            
            return True

        return batch_check_dets(w)

    def is_valid_addition(current_indices, p_idx):
        """
        Checks if adding p_idx to current_indices is valid.
        """
        m = len(current_indices)
        if m < 4:
            return True
        
        w = get_validity_matrix(p_idx, np.array(current_indices))
        return check_all_dets(w)

    def solve():
        start_time = time.time()
        deadline = start_time + 580 # Safety margin
        
        best_points = np.empty((0, 3), dtype=np.int64)
        best_size = 0
        
        np.random.seed(42)
        
        # Phase 1: Get a good initial solution using Greedy
        # We will run a few greedy passes to find a strong starting point
        initial_set = None
        for _ in range(5):
            order = np.random.permutation(num_points)
            current_indices = []
            for p_idx in order:
                if is_valid_addition(current_indices, p_idx):
                    current_indices.append(p_idx)
            if initial_set is None or len(current_indices) > len(initial_set):
                initial_set = current_indices
        
        if initial_set is None:
            return best_points
            
        current_indices = initial_set
        current_size = len(current_indices)
        
        if current_size > best_size:
            best_size = current_size
            best_points = grid[np.array(current_indices)]
            
        # Phase 2: Simulated Annealing / Local Search
        # State: set of indices
        # Move: 
        #   1. Try to add a random unselected point. If valid, accept.
        #   2. If invalid, identify a "conflict" set (4 points in current set that form a sphere with the new point).
        #      Remove one point from the conflict set (or a random point from current set) and try to add the new point.
        #      This is a "swap" move.
        #   3. Randomly remove a point and try to add a different one.
        
        # To make this efficient, we need to find the conflict quickly.
        # The check_all_dets function currently returns bool.
        # We need a version that returns the violating indices if any.
        
        def find_violation(current_indices, p_idx):
            """
            Returns None if valid.
            Returns the 4 indices (from current_indices) that form a sphere with p_idx if invalid.
            """
            m = len(current_indices)
            if m < 4:
                return None
                
            w = get_validity_matrix(p_idx, np.array(current_indices))
            
            # We need to find the specific combo.
            # We can reuse the batch logic but stop at the first violation.
            
            batch_size = 1000
            combos = itertools.combinations(range(m), 4)
            batch = []
            for c in combos:
                batch.append(c)
                if len(batch) >= batch_size:
                    idx = np.array(batch, dtype=np.int32)
                    mats = w[idx]
                    dets = np.linalg.det(mats)
                    viol_idx = np.where(np.abs(dets) < 0.5)[0]
                    if len(viol_idx) > 0:
                        # Return the local indices of the first violation
                        return list(batch[viol_idx[0]])
                    batch = []
            
            if batch:
                idx = np.array(batch, dtype=np.int32)
                mats = w[idx]
                dets = np.linalg.det(mats)
                viol_idx = np.where(np.abs(dets) < 0.5)[0]
                if len(viol_idx) > 0:
                    return list(batch[viol_idx[0]])
            
            return None

        # Annealing loop
        T = 10.0
        cooling_rate = 0.999
        max_steps = 100000
        
        step = 0
        while time.time() < deadline and step < max_steps:
            step += 1
            
            # Check time more frequently
            if step % 100 == 0 and time.time() > deadline - 10:
                break
                
            # Choose a move
            # 1. Try to add a random point from the unselected set
            selected_set = set(current_indices)
            unselected = [i for i in range(num_points) if i not in selected_set]
            
            if not unselected:
                # No unselected points, try swaps
                # Pick a random point to remove
                rem_idx = np.random.choice(current_indices)
                new_indices = [i for i in current_indices if i != rem_idx]
                # Try to add a random point
                if unselected: # This is empty, so we can't add. 
                    # In this case, we can only remove. But removing reduces size.
                    # We should avoid pure removals unless necessary for escape.
                    # For now, if no unselected, we are stuck unless we do a swap with a "better" point?
                    # But all points are selected. So we are at a local maximum of size.
                    # We can try to swap a point with another point to allow a future addition?
                    # This is complex. Let's just break if no unselected.
                    pass
                else:
                    break

            # Pick a random candidate to add
            cand = np.random.choice(unselected)
            
            # Check validity
            violation = find_violation(current_indices, cand)
            
            if violation is None:
                # Valid addition
                current_indices.append(cand)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_points = grid[np.array(current_indices)]
                # Reset temperature slightly? Or just continue.
                # Successful addition is always accepted.
            else:
                # Violation found.
                # Option A: Reject.
                # Option B: Swap. Remove one point from the violation set (or a random point) and try to add cand.
                # This is a "kick" move.
                
                # Probability of accepting a worse state (removal) is low, but we need to escape.
                # Actually, removing a point decreases size. We only want to remove if it allows a NET gain later?
                # Simulated Annealing usually accepts worse states with probability exp(-dE/T).
                # Here, "energy" is -size. So removing a point increases energy (bad).
                # We should only remove if we can immediately re-add something else?
                
                # Strategy: 
                # 1. Remove a random point from the violation set.
                # 2. Check if cand is now valid.
                # 3. If valid, add cand. Net change: 0 (removed 1, added 1). Size unchanged.
                #    This is a "neutral" move that changes the configuration.
                # 4. If invalid, try removing a different point?
                
                # Let's try a "swap" move:
                # Pick a point `rem` from current_indices.
                # Pick a point `add` from unselected.
                # If `add` is valid in (current - {rem}), then perform the swap.
                # This maintains size but explores the space.
                
                # To guide the search, we prefer swaps that resolve conflicts.
                # The violation set `violation` contains indices in `current_indices` (local indices).
                # Let's try removing one of these "conflicting" points.
                
                # Get the global index of the conflicting point
                # violation is a list of 4 local indices into current_indices
                # Let's pick one randomly
                local_conflict_idx = np.random.choice(violation)
                global_conflict_idx = current_indices[local_conflict_idx]
                
                # Try the swap: Remove global_conflict_idx, Add cand
                # New set: current_indices - {global_conflict_idx} + {cand}
                # We need to check if cand is valid against (current_indices - {global_conflict_idx})
                
                # Optimization: We already know cand violates the sphere defined by the 4 points in `violation`.
                # If we remove one of them, that specific sphere is broken.
                # But cand might violate ANOTHER sphere formed by the remaining points.
                # So we must re-check.
                
                new_indices = [i for i in current_indices if i != global_conflict_idx]
                
                # Check if cand is valid against new_indices
                # This is expensive. But we only do it for promising swaps.
                # We can do a quick check: 
                # If len(new_indices) < 4, it's valid.
                # Otherwise, check.
                
                # To save time, we can just try the swap and then verify the whole set?
                # No, we need to know if it's valid to accept.
                
                # Let's use the find_violation function again.
                new_violation = find_violation(new_indices, cand)
                
                if new_violation is None:
                    # Swap is valid!
                    # Replace global_conflict_idx with cand
                    current_indices = new_indices
                    current_indices.append(cand)
                    # Size unchanged.
                    # This helps escape local minima where a specific point blocks many others.
                else:
                    # Swap failed.
                    # Try a random swap instead?
                    # Or just reject.
                    # To avoid getting stuck, occasionally do a random swap.
                    if np.random.rand() < 0.1:
                        # Random swap
                        rem_idx = np.random.choice(current_indices)
                        add_idx = np.random.choice(unselected)
                        
                        new_indices = [i for i in current_indices if i != rem_idx]
                        new_violation = find_violation(new_indices, add_idx)
                        if new_violation is None:
                            current_indices = new_indices
                            current_indices.append(add_idx)
                            # Size unchanged
                            
            # Update best if size increased (handled in addition case)
            # If size unchanged, we don't update best_points unless we want to track diversity?
            # No, best_points is based on size.
            
            # Cooling
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
                
        return best_points

    return solve()

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
