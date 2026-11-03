import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4
    vectors_D = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            vectors_D.append(v)
            
    vectors_D = np.array(vectors_D, dtype=np.int8)
    
    # Compute conflict matrix for Type D
    # Conflict if dot product >= 3
    # Dot product range: -4 to 4
    # We need distance >= 2 => dist^2 >= 4
    # dist^2 = 4 + 4 - 2 dot = 8 - 2 dot
    # 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2
    # So conflict if dot > 2, i.e., dot >= 3
    
    # Optimization: Only compute upper triangle or use vectorized ops
    # 5280 x 5280 dot product
    # This might be memory heavy if not careful, but int8 -> int16/32 is fine.
    # 5280*5280*4 bytes ~ 110MB. Acceptable.
    
    dot_products = vectors_D @ vectors_D.T
    
    # Mask for conflicts (dot >= 3)
    # Note: Diagonal is 4, which is a conflict with itself, but we handle selection logic.
    conflict_mask = dot_products >= 3
    
    N_D = len(vectors_D)
    
    best_set_D = []
    
    # Greedy Independent Set with Random Restarts
    # We want to maximize the number of selected vectors.
    # Simple greedy: Pick random, remove neighbors.
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(range(N_D))
        current_set = []
        
        # Shuffle order for randomness
        # Instead of shuffling the whole list every time, we can pick randomly from available
        # But picking from a set is O(1) average.
        
        while available:
            # Pick a random element
            # Using next(iter(available)) is fast
            v_idx = next(iter(available))
            
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove neighbors
            # neighbors are indices j where conflict_mask[v_idx, j] is True
            # We can get the boolean array and use it to filter available
            # But filtering a set with a boolean array is slow.
            # Better: precompute neighbor lists.
            
            pass # Placeholder, will implement neighbor lists below

    # Precompute neighbor lists for speed
    # neighbor_list[i] = set of j such that conflict_mask[i, j]
    # This takes memory but speeds up the loop significantly.
    
    neighbor_lists = []
    for i in range(N_D):
        # Get indices where conflict_mask[i] is True
        # Exclude self? conflict_mask[i,i] is True.
        # We remove v_idx from available before removing neighbors, so self is already gone.
        # But we should not try to remove self again.
        # Let's just get the list of conflicting indices.
        conflicts = np.where(conflict_mask[i])[0]
        neighbor_lists.append(conflicts)
        
    # Re-run greedy with precomputed neighbors
    best_set_D = []
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(range(N_D))
        current_set = []
        
        while available:
            v_idx = next(iter(available))
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove all neighbors from available
            # conflicts is an array of indices
            # We need to remove these from the set 'available'
            # set.difference_update is efficient
            available.difference_update(neighbor_lists[v_idx])
            
        if len(current_set) > len(best_set_D):
            best_set_D = current_set
            
    # Combine Type A and selected Type D
    vectors_A = np.array(vectors_A, dtype=np.int64)
    vectors_D_selected = vectors_D[best_set_D].astype(np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_D_selected])
    
    return final_vectors
