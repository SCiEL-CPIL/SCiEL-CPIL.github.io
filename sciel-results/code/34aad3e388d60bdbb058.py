import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # 1. Generate candidate pool: Weight-4 vectors in R^11
    # These have norm^2 = 4.
    # The constraint is min_dist^2 >= max_norm^2.
    # If we restrict to weight-4 vectors, max_norm^2 = 4.
    # dist^2(u, v) = |u|^2 + |v|^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4  =>  2 u.v <= 4  =>  u.v <= 2.
    # For weight-4 vectors, u.v is the sum of products of overlapping coordinates.
    # If they share k indices, the dot product is sum of +/-1 for those k indices.
    # Possible dot products for weight-4 vectors:
    # - If disjoint (k=0): u.v = 0. (Valid, 0 <= 2)
    # - If share 1 (k=1): u.v in {-1, 1}. (Valid, 1 <= 2)
    # - If share 2 (k=2): u.v in {-2, 0, 2}. (Valid, 2 <= 2)
    # - If share 3 (k=3): u.v in {-3, -1, 1, 3}. (INVALID if u.v = 3 or -3? No, -3 <= 2 is true. 3 > 2 is FALSE.)
    #   Wait: u.v = 3 means they agree on all 3 shared signs. dist^2 = 8 - 6 = 2 < 4. Invalid.
    #   u.v = -3 means they disagree on all 3 shared signs. dist^2 = 8 - (-6) = 14 >= 4. Valid.
    #   u.v = 1 or -1. Valid.
    # - If share 4 (k=4): u.v in {-4, -2, 0, 2, 4}.
    #   u.v = 4 (identical vector): dist^2 = 0 < 4. Invalid (but we don't add duplicates).
    #   u.v = 2: dist^2 = 4 >= 4. Valid.
    #   u.v = 0: Valid.
    #   u.v = -2: Valid.
    #   u.v = -4: Valid.
    
    # So the ONLY invalid pairs among weight-4 vectors are those with u.v = 3.
    # u.v = 3 occurs if and only if they share exactly 3 indices and have the SAME signs on all 3.
    
    # This is a huge simplification! We just need to avoid pairs that share 3 indices with identical signs.
    
    # Generate all weight-4 vectors
    candidates = []
    indices = range(D)
    # We can restrict to "canonical" signs to reduce symmetry? 
    # No, signs matter for the dot product.
    # However, we can just generate all and let the algorithm pick.
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280.
    
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    n_cands = len(candidates)
    
    # Precompute dot products? 
    # 5280 x 5280 matrix of int8/int16 is ~55MB. 
    # But we only need to check if dot == 3.
    # Actually, we can check validity on the fly using vectorized operations against the current set.
    
    # Greedy Construction with Vectorized Conflict Check
    # State: current_set (list of indices into candidates)
    # To check if a new vector `v` can be added:
    #   Compute dot products of `v` with all vectors in `current_set`.
    #   Check if any dot product is 3.
    
    # To speed up, we can maintain the current set as a matrix.
    # But for greedy, we iterate through candidates in a random order and try to add them.
    
    # Better Strategy: Multi-start Greedy with Local Search
    # 1. Start with an empty set.
    # 2. Iterate through shuffled candidates.
    # 3. If candidate is compatible with current set (no dot=3), add it.
    # 4. Repeat this process multiple times with different shuffles.
    # 5. Use the best result as a seed for a more aggressive local search (swap-based).
    
    best_set = []
    best_count = 0
    
    # Optimization: Pre-compute "conflict masks" is hard because it's pairwise.
    # Instead, we can use the fact that conflicts are local (share 3 indices).
    
    # Let's do a few greedy passes first to get a good baseline.
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        order = np.random.permutation(n_cands)
        current_set = []
        
        for idx in order:
            v = candidates[idx]
            # Check against current set
            if len(current_set) > 0:
                # Vectorized dot product
                # current_set_vecs is a subset of candidates
                # We can keep the actual vectors in a list for speed in small sets
                # Or use indices and lookup.
                # Let's use indices and lookup from candidates array.
                # dots = candidates[current_set] @ v
                # This creates a temporary array.
                
                # Alternative: Pre-compute a "signature" or use the specific property.
                # Property: Conflict iff share 3 indices with same sign.
                
                # Let's stick to vectorized dot product for simplicity and correctness.
                # 5280 vectors. If set size is ~100, this is 100 dots. Fast.
                
                # We can optimize by maintaining a list of vectors in the set.
                pass
            
            # Let's implement the check efficiently.
            # We will maintain `current_vecs` as a numpy array of shape (k, D).
            # Initially empty.
            
        # This loop structure is a bit messy. Let's refactor.
        
    # Refactored Greedy
    def greedy_pack(order):
        current_indices = []
        current_vecs = np.empty((0, D), dtype=np.int8)
        
        for idx in order:
            v = candidates[idx]
            if len(current_indices) == 0:
                current_indices.append(idx)
                current_vecs = np.vstack([current_vecs, v])
            else:
                # Check dot products
                # v is shape (D,), current_vecs is (k, D)
                # dots = current_vecs @ v
                dots = current_vecs.dot(v)
                # Conflict if any dot == 3
                if not np.any(dots == 3):
                    current_indices.append(idx)
                    current_vecs = np.vstack([current_vecs, v])
                    
        return current_indices, current_vecs
    
    # Run a few greedy passes
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        order = np.random.permutation(n_cands)
        s_idx, s_vecs = greedy_pack(order)
        if len(s_idx) > best_count:
            best_count = len(s_idx)
            best_set = s_idx
            best_vecs = s_vecs
            # Early stop if we hit a very high number?
            # Theoretical max for weight 4 kissing in d=11?
            # Kissing number in 11 is 554? No, that's for unit vectors.
            # Here we are packing weight-4 vectors with min dist 2 (norm 2).
            # This is equivalent to packing unit vectors with min angle such that cos(theta) <= 1/2?
            # dist^2 = 2 - 2cos(theta) >= 4 (norm^2 of weight 4 is 4, wait. Norm is 2. dist >= 2.
            # dist^2 >= 4.
            # 8 - 2 u.v >= 4 => u.v <= 2.
            # u.v = |u||v| cos(theta) = 4 cos(theta).
            # 4 cos(theta) <= 2 => cos(theta) <= 0.5 => theta >= 60 degrees.
            # So we are packing vectors with angular separation >= 60 degrees.
            # In 11D, how many such vectors?
            # For d=11, kissing number is 554? No, K(11) is not known exactly but bounded.
            # Wait, the vectors are integer. This is a subset of the kissing number problem.
            # The "kissing number" for integer vectors might be smaller.
            # However, 16 per support cluster * (11 choose 4) / 4?
            # Let's just search.
            
    # Local Search: Try to add more or swap to improve?
    # Since greedy is near-optimal for this specific "avoid dot=3" constraint,
    # simple swaps might not help much if greedy already found a maximal set.
    # But different shuffles yield different maximal sets.
    
    # Let's run more greedy passes if time permits.
    # Each pass takes O(N * K * D) where N is pool size, K is set size.
    # 5280 * 100 * 11 ~ 6M ops. Very fast.
    
    iterations = 0
    while time.time() < deadline - 5:
        order = np.random.permutation(n_cands)
        s_idx, s_vecs = greedy_pack(order)
        if len(s_idx) > best_count:
            best_count = len(s_idx)
            best_set = s_idx
            best_vecs = s_vecs
        iterations += 1
        
        # Optional: If we find a very large set, try to perturb it?
        # For now, just keep finding new random maximal sets.
        
    # Return the best set found
    if len(best_set) == 0:
        # Fallback: Return a single valid vector
        return candidates[:1]
        
    result = candidates[best_set]
    return result

# EVOLVE-BLOCK-END
