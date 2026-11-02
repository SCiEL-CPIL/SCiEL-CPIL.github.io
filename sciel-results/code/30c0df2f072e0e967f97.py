import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Generate all integer vectors with exactly 4 non-zero entries of value +/- 1
    # Squared norm is 4.
    # We need a set S such that min pairwise distance >= max norm.
    # If we use only these vectors, max norm is 2.
    # We need min pairwise distance >= 2, i.e., squared distance >= 4.
    
    # 1. Generate candidates
    candidates = []
    indices_list = list(itertools.combinations(range(D), 4))
    
    for idx_tuple in indices_list:
        i, j, k, l = idx_tuple
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    if not candidates:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v])

    num_candidates = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # 2. Compute Conflict Graph
    # Conflict if squared distance < 4
    # dist_sq = ||u - v||^2
    # We can compute this using matrix multiplication or broadcasting.
    # Broadcasting 5280x11 x 11x5280 might be memory heavy?
    # 5280 * 11 * 4 bytes = ~232 KB.
    # Pairwise differences: 5280 x 5280 x 11.
    # 5280^2 * 11 * 4 bytes = ~1.2 GB. Too large.
    
    # Instead, we can iterate and compute neighbors on the fly or in batches.
    # Or use the property:
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # Conflict if 8 - 2 u.v < 4  =>  4 < 2 u.v  =>  u.v > 2.
    # Since u, v are +/- 1 in 4 positions, u.v is sum of products of 4 pairs.
    # Each product is +/- 1.
    # Sum can be -4, -2, 0, 2, 4.
    # u.v > 2 means u.v = 4.
    # u.v = 4 implies u = v.
    # Wait.
    # If u and v have the same 4 indices and same signs, u.v = 4.
    # If they differ in 1 sign, u.v = 2.
    # If they differ in 2 signs, u.v = 0.
    # If they differ in 3 signs, u.v = -2.
    # If they differ in 4 signs, u.v = -4.
    
    # What if they have different indices?
    # u = e1+e2+e3+e4.
    # v = e1+e2+e3+e5.
    # u.v = 1+1+1+0 = 3.
    # Dist sq = 8 - 2(3) = 2.
    # Conflict!
    
    # So conflict condition: u.v >= 3.
    # Possible values of u.v for distinct vectors:
    # Same indices, 1 sign diff: 2. (Dist sq 4, OK)
    # Same indices, 2 sign diff: 0. (Dist sq 8, OK)
    # 3 shared indices, 1 diff: 3. (Dist sq 2, CONFLICT)
    # 2 shared indices, 2 diff: 2. (Dist sq 4, OK)
    # 1 shared index, 3 diff: 1. (Dist sq 6, OK)
    # 0 shared indices: 0. (Dist sq 8, OK)
    
    # So the ONLY conflicts are between vectors that share exactly 3 indices (and have same signs on those 3).
    # Or vectors that are identical (u.v=4).
    
    # This simplifies the graph immensely!
    # We don't need to check all pairs.
    # For each vector u, its neighbors are:
    # 1. v with same 4 indices, 1 sign flipped? No, u.v=2, dist sq 4. Not a conflict.
    # 2. v with 3 shared indices (same signs), 1 different index.
    #    Number of such v:
    #    Choose which 3 indices to keep: 4 ways.
    #    Choose the new index: 11 - 4 = 7 ways.
    #    Signs on the 3 kept indices must be same as u.
    #    Sign on the new index can be +/- 1. (2 ways).
    #    Total neighbors per u: 4 * 7 * 2 = 56.
    
    # Let's verify.
    # u = (1, 1, 1, 1) at idx 0,1,2,3.
    # v keeps 0,1,2. New idx 4.
    # v = (1, 1, 1, 0, 1, 0...) or (1, 1, 1, 0, -1, 0...).
    # u.v = 1+1+1+0 = 3.
    # Dist sq = 8 - 6 = 2. Conflict.
    
    # So degree is 56.
    # We need to find a Max Independent Set in a graph where each node has 56 neighbors.
    # N = 5280.
    # Greedy MIS should work well.
    
    # Build adjacency list
    # Map from vector tuple to index
    vec_to_idx = {}
    for i, v in enumerate(candidates):
        vec_to_idx[tuple(v)] = i
        
    adj = [[] for _ in range(num_candidates)]
    
    # To build adj efficiently, we can iterate over all candidates and generate their neighbors.
    for i, v in enumerate(candidates):
        # Get indices of non-zeros
        non_zero_idx = np.nonzero(v)[0]
        # non_zero_idx has 4 elements
        
        # Generate neighbors
        # Keep 3, change 1
        for keep_mask in range(4):
            # keep_mask is the index in non_zero_idx to DROP
            keep_indices = [non_zero_idx[j] for j in range(4) if j != keep_mask]
            drop_index = non_zero_idx[keep_mask]
            
            # Signs of kept indices
            kept_signs = [v[idx] for idx in keep_indices]
            
            # Try all new indices
            for new_idx in range(D):
                if new_idx == drop_index:
                    continue # This would be the same set of indices, just different sign?
                    # If new_idx is one of the kept indices, it's not a "new" index in the sense of changing the support.
                    # We are looking for v with 3 shared indices.
                    # The 4th index of v must be different from the 4 indices of u?
                    # No, v has 4 non-zeros.
                    # If v shares 3 with u, the 4th index of v is NOT in the 3 shared.
                    # It could be the dropped index of u?
                    # If v uses the dropped index of u, then v has the SAME 4 indices as u.
                    # If v has the same 4 indices, and shares 3 signs, and 1 sign is different:
                    # u.v = 2. Dist sq 4. Not a conflict.
                    # So we only care about v with a DIFFERENT 4th index.
                    
                    # So new_idx must NOT be in keep_indices.
                    # And new_idx should not be drop_index if we want a different support?
                    # If new_idx == drop_index, the support is the same.
                    # We established same support with 1 sign diff is NOT a conflict.
                    # So we can skip new_idx == drop_index.
                    
                if new_idx in keep_indices:
                    continue # new_idx must be outside the kept set to form a new support?
                    # Actually, v has 4 non-zeros.
                    # If new_idx is in keep_indices, then v has only 3 non-zeros? No.
                    # v is defined by 4 indices.
                    # If we keep 3 indices from u, and add new_idx, the support of v is {keep_indices} U {new_idx}.
                    # If new_idx is in keep_indices, the size is 3. Invalid.
                    # So new_idx must be outside keep_indices.
                    
                    # We also know new_idx != drop_index (handled above).
                    
                # So new_idx is any index in 0..10 except the 4 indices of u.
                
                # Construct v
                v_new = np.zeros(D, dtype=np.int64)
                for idx, sign in zip(keep_indices, kept_signs):
                    v_new[idx] = sign
                # Two possible signs for new_idx
                for s in (1, -1):
                    v_new[new_idx] = s
                    key = tuple(v_new)
                    if key in vec_to_idx:
                        j = vec_to_idx[key]
                        if i != j:
                            adj[i].append(j)
                            # adj[j].append(i) # We can do this later or just use directed
                            
    # Make adjacency symmetric
    # Actually, the generation above finds all neighbors for each i.
    # We can just use adj[i] for greedy.
    
    # Greedy Independent Set
    available = set(range(num_candidates))
    independent_set = []
    
    # Sort by degree?
    # Degrees are roughly uniform (56).
    # Random order might be better to avoid bias?
    # Or just iterate.
    
    # To improve, we can do multiple restarts with random shuffling of "pick order"
    # But simple greedy is a good start.
    
    # Let's try a greedy that picks the node with the fewest available neighbors?
    # That's expensive to update.
    # Simple: Pick first available, remove it and its neighbors.
    
    # To get a better result, we can try random restarts.
    best_set = []
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        available = set(range(num_candidates))
        current_set = []
        
        # Randomize the order of selection
        # We can't shuffle the set easily, but we can pick a random available node each time?
        # No, that's slow.
        # We can create a list of available nodes, shuffle it, and iterate.
        # But as we remove nodes, the list becomes stale.
        
        # Alternative:
        # Use a list of indices.
        # Maintain a boolean array 'is_available'.
        # Iterate through a shuffled list of indices.
        
        indices = list(range(num_candidates))
        np.random.shuffle(indices)
        
        is_available = np.ones(num_candidates, dtype=bool)
        
        for idx in indices:
            if not is_available[idx]:
                continue
                
            # Add idx to set
            current_set.append(idx)
            is_available[idx] = False
            
            # Remove neighbors
            for neighbor in adj[idx]:
                if is_available[neighbor]:
                    is_available[neighbor] = False
                    
        if len(current_set) > len(best_set):
            best_set = current_set
            
    if not best_set:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v])
        
    result = candidates[best_set]
    return result
