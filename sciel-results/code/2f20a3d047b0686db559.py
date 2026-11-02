import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Generate candidates
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i (Norm sq 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Three 1s, rest 0: ±e_i ± e_j ± e_k (Norm sq 3)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix M
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    R_sq = 4
    
    # Threshold matrix: Max allowed dot product for pair (i,j)
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    Thresh = num // 2
    
    # Precompute conflict matrix
    conflict_matrix = M > Thresh
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy A: Random orderings with vectorized compatibility check
    for trial in range(100):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_indices = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if len(selected_indices) == 0:
                selected_indices.append(idx)
                selected_mask[idx] = True
                continue
            
            # Check if idx is compatible with all selected
            # conflict_matrix[idx, selected_mask] gives conflicts with selected set
            if not np.any(conflict_matrix[idx, selected_mask]):
                selected_indices.append(idx)
                selected_mask[idx] = True
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Greedy with "least conflicts" priority
    if time.time() < deadline - 20:
        conflict_counts = np.sum(conflict_matrix, axis=1) - 1
        
        order_b = np.argsort(conflict_counts, kind='stable')
        
        selected_indices = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order_b:
            if len(selected_indices) == 0:
                selected_indices.append(idx)
                selected_mask[idx] = True
                continue
            
            if not np.any(conflict_matrix[idx, selected_mask]):
                selected_indices.append(idx)
                selected_mask[idx] = True
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy C: Greedy with "most central" (highest degree in compatibility graph)
    if time.time() < deadline - 10:
        compat_counts = N - 1 - conflict_counts
        order_c = np.argsort(-compat_counts, kind='stable')
        
        selected_indices = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order_c:
            if len(selected_indices) == 0:
                selected_indices.append(idx)
                selected_mask[idx] = True
                continue
            
            if not np.any(conflict_matrix[idx, selected_mask]):
                selected_indices.append(idx)
                selected_mask[idx] = True
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy D: Local search - try swapping or adding
    if best_set is not None and time.time() < deadline - 5:
        # First, try to add any unselected vectors
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            if not np.any(conflict_matrix[idx, best_set]):
                best_set[idx] = True
                best_count += 1
        
        # Then try local swaps: remove one, add two
        if time.time() < deadline - 3:
            selected_indices = np.where(best_set)[0]
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in selected_indices:
                    if time.time() > deadline - 3:
                        break
                    # Try removing i and adding two others
                    best_set[i] = False
                    temp_count = best_count - 1
                    
                    # Find candidates that can be added
                    addable = []
                    for j in unselected:
                        if not np.any(conflict_matrix[j, best_set]):
                            addable.append(j)
                    
                    # Try adding pairs from addable
                    for a_idx in range(min(len(addable), 50)):
                        for b_idx in range(a_idx + 1, min(len(addable), 50)):
                            a = addable[a_idx]
                            b = addable[b_idx]
                            if not conflict_matrix[a, b]:
                                best_set[a] = True
                                best_set[b] = True
                                temp_count += 2
                                if temp_count > best_count:
                                    best_count = temp_count
                                    improved = True
                                best_set[a] = False
                                best_set[b] = False
                                break
                        if improved:
                            break
                    
                    best_set[i] = True
                    if improved:
                        break
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
