"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n
    
    # 1. Precompute comparability matrix
    # comp[i, j] = 1 if i and j are comparable (one strict subset of other)
    masks = np.arange(num_masks, dtype=np.int32)
    # To save memory and time, we can compute on the fly or use a smaller representation?
    # 4096x4096 int8 is 16MB. This is fine.
    
    # Efficient construction of comp
    # i is subset of j iff (i | j) == j
    # i != j
    # We can do this in blocks to avoid huge temporary arrays if needed, but 16MB is small.
    
    # However, vectorized ops on 4096x4096 might be slow if done repeatedly.
    # We only do it once.
    
    # Using broadcasting
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    # 2. Greedy Initialization
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    # Greedy: pick set with max degree, then next max degree to selected, etc.
    # To speed up, we can just pick random or simple greedy.
    # Simple greedy:
    scores = comp.sum(axis=1).astype(np.int32)
    
    for _ in range(m):
        if time.time() > deadline - 10:
            break
        # Pick argmax of scores, among not selected
        # To do this efficiently, we can just find max in scores where not selected
        # But scores changes? No, in greedy init, we don't update scores dynamically in the most optimal way (which would be O(N^2)), 
        # but simple greedy is fine for initialization.
        # Actually, let's do a slightly better greedy:
        # Calculate marginal gain for each unselected set.
        
        if len(selected_indices) == 0:
            best_idx = int(np.argmax(scores))
        else:
            # Marginal gain for x = sum(comp[x, k] for k in selected_indices)
            # This is O(N * m)
            # We can compute this vectorized
            # comp[:, selected_indices].sum(axis=1)
            gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
            gains[selected] = -1
            best_idx = int(np.argmax(gains))
            
        selected_indices.append(best_idx)
        selected[best_idx] = True
        
    # 3. Local Search (Best Improvement Swap)
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    # score_in_set[x] = sum(comp[x, k] for k in current_set)
    # Initialize
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    best_score = 0
    # Calculate initial score
    # Sum of comp[i, j] for i < j in set
    # Or sum(score_in_set[i] for i in set) / 2
    for idx in selected_indices:
        best_score += score_in_set[idx]
    best_score //= 2
    
    current_score = best_score
    best_indices = list(selected_indices)
    
    # Single-swap local search
    while time.time() < deadline - 5:
        # Find best swap
        best_delta = 0
        best_out = -1
        best_in = -1
        
        # Precompute masked scores once per iteration (independent of A)
        masked_scores = np.where(in_set, -1, score_in_set)
        max_val = int(np.max(masked_scores))
        max_idx = int(np.argmax(masked_scores))
        
        # Find the A in set with minimum score_in_set[A]
        # delta = max_val - min_A score_in_set[A]
        min_A_val = min(score_in_set[A] for A in selected_indices)
        
        if max_val - min_A_val > best_delta:
            best_delta = max_val - min_A_val
            # Find which A has min score
            best_out = min(selected_indices, key=lambda x: score_in_set[x])
            best_in = max_idx
        
        if best_delta <= 0:
            break # No improvement possible
            
        # Apply swap
        score_in_set += comp[:, best_in] - comp[:, best_out]
        
        selected_indices[selected_indices.index(best_out)] = best_in
        current_set.remove(best_out)
        current_set.add(best_in)
        in_set[best_out] = False
        in_set[best_in] = True
        
        current_score += best_delta
        if current_score > best_score:
            best_score = current_score
            best_indices = list(selected_indices)
    
    # 2-swap local search: remove two, add two
    if time.time() < deadline - 5:
        while time.time() < deadline - 5:
            best_delta2 = 0
            best_out1 = -1
            best_out2 = -1
            best_in1 = -1
            best_in2 = -1
            
            # For each pair (A, B) in set, find best pair (C, D) not in set
            # delta = (score_in_set[C] + score_in_set[D] - comp[C,D]) - (score_in_set[A] + score_in_set[B] - comp[A,B])
            # Wait, that's not quite right. Let me think more carefully.
            # When we remove A and B and add C and D:
            # New score = old_score - (contributions from A and B to others in set) + (contributions from C and D to others in set)
            # But also we lose comp[A,B] and gain comp[C,D]
            # Actually, let's use the score_in_set formulation.
            # score_in_set[x] = sum of comp[x, k] for k in current_set
            # Total score = sum(score_in_set[x] for x in set) / 2
            
            # If we remove A and B:
            # New score_in_set for remaining = score_in_set - comp[:,A] - comp[:,B]
            # Then add C and D:
            # New score_in_set = (score_in_set - comp[:,A] - comp[:,B]) + comp[:,C] + comp[:,D]
            # New total = sum of new_score_in_set over new_set / 2
            
            # This is complex. Let me use a simpler approach:
            # delta = (comp[C,D] + sum_{k in set\{A,B}} (comp[C,k]+comp[D,k])) - (comp[A,B] + sum_{k in set\{A,B}} (comp[A,k]+comp[B,k]))
            
            # For efficiency, let's iterate over pairs in set and for each, find best pair not in set.
            # This is O(m^2 * N^2) which is too slow.
            
            # Alternative: for each C not in set, compute gain_C = sum_{k in set} comp[C,k]
            # Then for each D not in set, D != C, compute gain_CD = gain_C + gain_D - comp[C,D] - (sum of comp for the two removed)
            
            # This is still complex. Let me skip 2-swap for now and focus on restarts.
            break  # Skip 2-swap for simplicity
    
    # Random restarts
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        np.random.seed(restart * 12345 + 67890)
        selected_indices = list(np.random.choice(num_masks, size=m, replace=False))
        current_set = set(selected_indices)
        in_set = np.zeros(num_masks, dtype=bool)
        for idx in selected_indices:
            in_set[idx] = True
        
        score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
        
        restart_score = 0
        for idx in selected_indices:
            restart_score += score_in_set[idx]
        restart_score //= 2
        
        # Single-swap local search
        while time.time() < deadline - 5:
            masked_scores = np.where(in_set, -1, score_in_set)
            max_val = int(np.max(masked_scores))
            max_idx = int(np.argmax(masked_scores))
            
            min_A_val = min(score_in_set[A] for A in selected_indices)
            
            if max_val - min_A_val <= 0:
                break
            
            best_delta = max_val - min_A_val
            best_out = min(selected_indices, key=lambda x: score_in_set[x])
            best_in = max_idx
            
            score_in_set += comp[:, best_in] - comp[:, best_out]
            
            selected_indices[selected_indices.index(best_out)] = best_in
            current_set.remove(best_out)
            current_set.add(best_in)
            in_set[best_out] = False
            in_set[best_in] = True
            
            restart_score += best_delta
            if restart_score > best_score:
                best_score = restart_score
                best_indices = list(selected_indices)
            
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
