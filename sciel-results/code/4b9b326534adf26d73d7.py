import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
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
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix M
    M = candidates @ candidates.T
    
    # Conflict if dot product > 2
    # Because dist^2 = 4 + 4 - 2*dot >= 4 => dot <= 2
    conflict = M > 2
    
    best_count = 0
    best_set = None
    
    rng = np.random.default_rng(42)
    
    # Vectorized Greedy with Min-Degree Heuristic
    # We run multiple trials with different random orderings or just deterministic min-degree
    # Min-degree is usually very strong for packing problems.
    
    for trial in range(200):
        if time.time() > deadline:
            break
            
        # For trial 0, use natural order. For others, shuffle?
        # Min-degree doesn't depend on order if we always pick the best available.
        # But if there are ties, the order matters.
        # Let's randomize the tie-breaking or the initial order.
        
        remaining = np.arange(N)
        selected_mask = np.zeros(N, dtype=bool)
        
        # To break ties randomly, we can add small random noise to conflict counts?
        # Or just shuffle `remaining` initially?
        # If we shuffle initially, the min-degree picks will be different.
        
        if trial > 0:
            remaining = rng.permutation(remaining)
            
        while len(remaining) > 0:
            # Find candidates in `remaining` that are compatible with all selected
            # conflict[remaining, :] is (len(remaining), N)
            # We only care about columns where selected_mask is True
            if np.any(selected_mask):
                compatible_mask = np.all(~conflict[remaining][:, selected_mask], axis=1)
            else:
                compatible_mask = np.ones(len(remaining), dtype=bool)
                
            avail_indices_local = np.where(compatible_mask)[0]
            
            if len(avail_indices_local) == 0:
                break
                
            avail_indices_global = remaining[avail_indices_local]
            
            if len(avail_indices_global) == 1:
                pick = avail_indices_global[0]
            else:
                # Heuristic: Pick the one with the fewest conflicts with the rest of `remaining`
                # This helps preserve options for future steps.
                # conflict[avail_indices_global, remaining]
                # Sum along axis 1 (count conflicts for each candidate with all remaining)
                # We exclude the candidate itself? conflict[i,i] is False (0), so it's fine.
                conflict_counts = np.sum(conflict[avail_indices_global][:, remaining], axis=1)
                
                # To break ties randomly, we can pick randomly among the minimums
                min_count = np.min(conflict_counts)
                min_indices = np.where(conflict_counts == min_count)[0]
                
                if len(min_indices) > 1:
                    pick_idx = min_indices[rng.integers(0, len(min_indices))]
                else:
                    pick_idx = min_indices[0]
                    
                pick = avail_indices_global[pick_idx]
                
            selected_mask[pick] = True
            remaining = remaining[remaining != pick]
            
        count = np.sum(selected_mask)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
            
        # If we reached the theoretical max, we can stop
        if best_count >= 275:
            break

    if best_set is None:
        # Fallback: Return Type 1 vectors
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
