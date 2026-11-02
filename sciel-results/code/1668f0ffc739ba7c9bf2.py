import time
import numpy as np
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate candidate vectors
    # We look for vectors with small norms.
    # Max norm squared to consider initially.
    # If we use unit vectors, max norm sq = 1.
    # If we mix in norm sq = 2, max norm sq = 2.
    
    # Let's generate vectors with entries in {-1, 0, 1} first (norm sq <= 11)
    # And maybe {-2, -1, 0, 1, 2} for some components.
    # To keep it fast, let's stick to {-1, 0, 1} and {-2, -1, 0, 1, 2} for small support.
    
    candidates = set()
    
    # Add unit vectors (norm sq 1)
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.add(tuple(v))
            
    # Add weight 2 vectors (norm sq 2 or 4)
    # e.g., (1, 1, 0...), (1, -1, 0...), (2, 0, 0...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.add(tuple(v))
                
    # Add weight 1 with value 2 (norm sq 4)
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            candidates.add(tuple(v))

    # Convert to list of numpy arrays
    cand_list = np.array(list(candidates), dtype=np.int64)
    
    # Precompute norms squared
    norms_sq = np.sum(cand_list**2, axis=1)
    
    # Sort candidates by norm squared, then index for determinism
    # We want to try smaller norms first to keep R low?
    # Actually, the search will decide.
    order = np.argsort(norms_sq)
    cand_list = cand_list[order]
    norms_sq = norms_sq[order]
    
    # Initial best set: Unit vectors
    # Indices of unit vectors in cand_list
    unit_indices = np.where(norms_sq == 1)[0]
    
    best_set = [cand_list[i] for i in unit_indices]
    best_size = len(best_set)
    
    # Helper to check validity of a set of vectors
    # Returns True if valid, and the max norm squared
    def is_valid(set_vecs):
        if not set_vecs:
            return True, 0
        arr = np.array(set_vecs)
        n = len(arr)
        # Max norm squared
        max_norm_sq = np.max(np.sum(arr**2, axis=1))
        
        # Check pairwise distances
        # dist^2(i,j) = ||vi||^2 + ||vj||^2 - 2 vi.vj
        # We need dist^2(i,j) >= max_norm_sq for all i != j
        
        # Compute all pairwise dot products
        # arr @ arr.T
        dots = arr @ arr.T
        norms_sq_arr = np.sum(arr**2, axis=1)
        
        # dist_sq[i,j] = norms_sq_arr[i] + norms_sq_arr[j] - 2*dots[i,j]
        # We can compute this for the whole matrix
        dist_sq = norms_sq_arr[:, None] + norms_sq_arr[None, :] - 2 * dots
        
        # Zero out diagonal (distance to self is 0, which is < max_norm_sq usually, but we ignore i==j)
        np.fill_diagonal(dist_sq, max_norm_sq + 1) # Ensure diagonal doesn't fail check
        
        # Check if all off-diagonal entries are >= max_norm_sq
        if np.min(dist_sq) < max_norm_sq:
            return False, max_norm_sq
        return True, max_norm_sq

    # Local Search
    # We will perform random swaps
    # State: list of indices into cand_list? Or list of vectors?
    # Using vectors is easier for logic, but indices are faster for lookup.
    # Let's use indices.
    
    current_indices = list(unit_indices)
    
    # Precompute norms for quick access
    # We need to check validity of a set of indices.
    
    def check_valid_indices(indices):
        if not indices:
            return True
        arr = cand_list[indices]
        n = len(arr)
        max_norm_sq = np.max(norms_sq[indices])
        
        if n == 1:
            return True
            
        dots = arr @ arr.T
        norms_sq_arr = norms_sq[indices]
        dist_sq = norms_sq_arr[:, None] + norms_sq_arr[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, max_norm_sq + 1)
        
        return np.min(dist_sq) >= max_norm_sq

    # Initial check
    if not check_valid_indices(current_indices):
         # Should not happen for unit vectors
         print("Initial set invalid?")
         return np.array(best_set)

    best_indices = list(current_indices)
    best_size = len(current_indices)
    
    # Search Loop
    # Strategy: Try to add vectors first (growth), if fail, try swap (move)
    # To improve efficiency, we can precompute dot products for candidates against current set?
    # But the set changes.
    
    # Better strategy:
    # 1. Try adding a random candidate. If valid, keep it.
    # 2. If adding fails (or after some successful adds), try swapping: remove one, add another.
    # 3. Also try removing one if it helps? (Unlikely to help size, but might allow future adds)
    
    # Let's implement a more directed search.
    # We maintain a set of "active" candidates that we try to add.
    # Actually, random search is often good for these.
    
    # Improvement:
    # When checking validity of adding 'idx_add' to 'current_indices',
    # we only need to check the distances from 'idx_add' to all vectors in 'current_indices'.
    # We don't need to re-check pairwise distances within 'current_indices' if they were already valid.
    # This reduces complexity from O(N^2) to O(N) per check.
    
    def check_add_valid(current_indices, idx_add):
        # current_indices is a list of indices
        # idx_add is the new index to add
        # Check if adding idx_add to current_indices keeps the set valid.
        # Assumption: current_indices is already a valid set.
        
        if not current_indices:
            return True
        
        # Get the vector to add
        v_add = cand_list[idx_add]
        norm_add_sq = norms_sq[idx_add]
        
        # The new max norm squared will be max(old_max, norm_add_sq)
        # We need to know the old max norm squared of current_indices.
        # Let's compute it.
        old_norms = norms_sq[current_indices]
        old_max_norm_sq = np.max(old_norms)
        
        new_max_norm_sq = max(old_max_norm_sq, norm_add_sq)
        
        # Check distances from v_add to all v in current_indices
        # dist^2 = ||v_add||^2 + ||v||^2 - 2 v_add . v
        # We need dist^2 >= new_max_norm_sq
        
        # Compute dot products of v_add with all vectors in current_indices
        # v_add is shape (D,), cand_list[current_indices] is shape (N, D)
        dots = cand_list[current_indices] @ v_add
        
        # dist_sq for each pair
        dist_sqs = norm_add_sq + old_norms - 2 * dots
        
        # Check if all are >= new_max_norm_sq
        if np.min(dist_sqs) < new_max_norm_sq:
            return False
            
        # Also, we must ensure that the existing pairs in current_indices are still valid with the new max norm.
        # If new_max_norm_sq > old_max_norm_sq, the constraint becomes tighter for existing pairs.
        # If new_max_norm_sq == old_max_norm_sq, existing pairs are already valid.
        
        if new_max_norm_sq > old_max_norm_sq:
            # We need to re-check the whole set?
            # Or just check if min_dist of current set is >= new_max_norm_sq.
            # Since current set was valid with old_max_norm_sq, and new_max_norm_sq is larger,
            # it's possible that some pair in current set has distance < new_max_norm_sq.
            # So we must check the whole set if the max norm increases.
            
            # To avoid full check every time, we can store the current min_dist_sq of the set.
            # But maintaining that is complex.
            # For now, if max norm increases, do a full check.
            # This happens when we add a vector with larger norm than any existing.
            
            # Let's do a full check in this case.
            test_indices = current_indices + [idx_add]
            return check_valid_indices(test_indices)
        else:
            # Max norm didn't increase, so existing pairs are still valid.
            return True

    # We need to track the current max norm squared to use in check_add_valid efficiently?
    # Actually, check_add_valid computes it.
    
    # Let's update the search loop to use this faster check.
    
    # We also need to handle the case where we remove a vector.
    # Removing a vector always keeps the set valid (subset of valid set is valid).
    # So we can remove freely.
    
    # Search Loop
    while time.time() < deadline:
        # 1. Try to add a random candidate
        # Pick a random candidate not in current set
        # To make this efficient, we can pick from all candidates and check if it's in current set.
        # If it is, skip or try another.
        
        # To bias towards adding vectors that might fit, we can try candidates with smaller norms first?
        # Or just random.
        
        # Let's try adding.
        idx_add = np.random.randint(len(cand_list))
        
        if idx_add not in current_indices:
            if check_add_valid(current_indices, idx_add):
                current_indices.append(idx_add)
                if len(current_indices) > best_size:
                    best_size = len(current_indices)
                    best_indices = list(current_indices)
            else:
                # Add failed. Try swapping?
                # Remove a random vector and try to add a different one?
                # Or just remove a random vector to allow future adds?
                # Removing a vector reduces size, which is bad for score, but might allow adding 2 new ones.
                
                # Let's try a swap: remove one, add one.
                if current_indices:
                    idx_remove = np.random.choice(current_indices)
                    new_indices = [i for i in current_indices if i != idx_remove]
                    
                    # Try to add a random candidate to the reduced set
                    idx_add2 = np.random.randint(len(cand_list))
                    if idx_add2 not in new_indices:
                        if check_add_valid(new_indices, idx_add2):
                            current_indices = new_indices + [idx_add2]
                            # Size didn't change, but configuration did.
                            # This might allow future adds.
                            # We don't update best_size here since size is same.
                            # But we should check if this configuration is better? No, score is just k.
                            # So only growth matters for best_size.
        
        # Occasionally, try to remove a vector to escape local optima?
        # If we are stuck (no adds for a while), removing might help.
        # But random swaps above do this.
        
        # Let's add a small probability of pure removal to reduce size and restart?
        # No, we want to maximize size.
        
        # To improve search, we can try to add multiple vectors in a row if possible.
        # The loop does this naturally if adds are successful.
        
        # We can also try to add specific candidates that are "close" to the current set?
        # Hard to define.
        
        # Let's try to add a candidate that has small dot products with the current set?
        # This would require precomputation or expensive checks.
        
        # For now, random is okay.
        
        # One optimization: if we have many vectors, check_add_valid is O(N).
        # N is growing.
        # We should limit the number of candidates we consider to add?
        # No, we want the best ones.
        
        # Let's see if we can improve the candidate set.
        # Currently we have:
        # - Unit vectors (22)
        # - Weight 2 vectors with entries +/-1 (110 * 4 = 440? No, combinations(11,2)=55, 4 signs = 220)
        # - Weight 1 vectors with entry +/-2 (22)
        # Total ~ 264 candidates.
        
        # We can add more candidates with norm sq 5, 8, etc.
        # E.g. (1, 1, 1, 0...) norm sq 3.
        # (1, 1, 1, 1, 0...) norm sq 4.
        
        # Let's generate more candidates in the next iteration of the code.
        # But for now, let's stick to this set and improve the search.
        
        # If we are stuck, we can try a "kick" by removing several vectors and re-adding.
        # This is complex.
        
        # Let's just run the loop.
        pass

    # Return best found
    result = cand_list[best_indices]
    return result
