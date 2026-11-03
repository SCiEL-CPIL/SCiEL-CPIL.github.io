import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph vectorized
    # Represent each support as a bitmask
    support_masks = np.zeros(N_S, dtype=np.uint64)
    for i, s in enumerate(supports):
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks[i] = mask
    
    # Conflict if intersection size >= 3
    # Use popcount on AND of masks - vectorized with lookup table
    # Precompute popcount for 16-bit chunks
    popcount_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
    
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        if time.time() > deadline - 80:
            break
        ands = support_masks[i] & support_masks
        # Split into 16-bit chunks for popcount lookup
        low = (ands & 0xFFFF).astype(np.int64)
        high = ((ands >> 16) & 0xFFFF).astype(np.int64)
        inter_counts = popcount_16[low] + popcount_16[high]
        conflict_supports[i, :] = inter_counts >= 3
        conflict_supports[:, i] = inter_counts >= 3
    
    # Symmetrize
    conflict_supports = np.maximum(conflict_supports, conflict_supports.T)
    np.fill_diagonal(conflict_supports, False)
    
    # Find maximum independent set using greedy + local search
    best_support_set = []
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Greedy initialization with random order
        order = np.arange(N_S)
        rng.shuffle(order)
        
        in_set = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        
        for v in order:
            if not blocked[v] and not in_set[v]:
                in_set[v] = True
                blocked |= conflict_supports[v]
                
        current_size = np.sum(in_set)
        
        # Local search: try single swaps (remove one, add one or two)
        improved = True
        while improved:
            if time.time() > deadline - 55:
                break
            improved = False
            in_set_indices = np.where(in_set)[0]
            for v_remove in in_set_indices:
                if time.time() > deadline - 55:
                    break
                # Temporarily remove v_remove
                in_set[v_remove] = False
                # Recompute blocked from scratch
                blocked = np.zeros(N_S, dtype=bool)
                for v in np.where(in_set)[0]:
                    blocked |= conflict_supports[v]
                
                # Find nodes not in set and not blocked
                candidates = ~in_set & ~blocked
                cand_indices = np.where(candidates)[0]
                
                # Try to add candidates greedily (up to 2)
                added_indices = []
                for c in cand_indices:
                    if len(added_indices) >= 2:
                        break
                    if not blocked[c]:
                        in_set[c] = True
                        blocked |= conflict_supports[c]
                        added_indices.append(c)
                
                new_size = np.sum(in_set)
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert properly
                    in_set[v_remove] = True
                    for c in added_indices:
                        in_set[c] = False
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = np.where(in_set)[0].tolist()
    
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Greedy fill using vectorized dot product checks
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Precompute matrix for current set
    current_arr = np.array(current_set_vectors) if current_set_vectors else np.zeros((0, D))
    
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)
    
    for i in indices_all:
        if time.time() > deadline - 20:
            break
            
        v = all_vectors_D[i]
        
        if tuple(v) in current_set_vectors_set:
            continue
            
        # Check compatibility: dot(v, u) <= 2 for all u in set
        if len(current_set_vectors) > 0:
            dots = current_arr @ v
            if np.any(dots >= 3):
                continue
        
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_arr = np.vstack([current_arr, v.reshape(1, -1)])
        
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Try adding Type B vectors (two non-zero ±1, norm²=2)
    vectors_B_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B_candidates.append(v)
    
    vectors_B_candidates = np.array(vectors_B_candidates, dtype=np.int64)
    N_B = len(vectors_B_candidates)
    
    added_B = []
    added_B_set = set()
    
    # Precompute dots of B candidates with A and D vectors
    dots_BA = vectors_B_candidates @ vectors_A.T  # (N_B, 22)
    dots_BD = vectors_B_candidates @ final_vectors_D.T  # (N_B, len(D))
    
    # B vs B: need dot = 0 (disjoint supports or orthogonal)
    dots_BB = vectors_B_candidates @ vectors_B_candidates.T  # (N_B, N_B)
    
    for i in range(N_B):
        if time.time() > deadline - 5:
            break
        
        vt = tuple(vectors_B_candidates[i])
        if vt in added_B_set:
            continue
        
        # Check against Type A: dot <= 1
        if np.any(dots_BA[i] >= 2):
            continue
        
        # Check against Type D: dot <= 1
        if np.any(dots_BD[i] >= 2):
            continue
        
        # Check against other Type B: dot = 0
        if added_B:
            added_B_arr = np.array(added_B)
            dots_with_added = added_B_arr @ vectors_B_candidates[i]
            if np.any(dots_with_added != 0):
                continue
        
        added_B.append(vectors_B_candidates[i])
        added_B_set.add(vt)
    
    if added_B:
        added_B = np.array(added_B, dtype=np.int64)
        final_vectors = np.vstack([vectors_A, final_vectors_D, added_B])
    else:
        final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
