import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Utility to count comparable pairs for validation or debugging.
    """
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            sj = sets[j]
            if si and sj:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = set(si)
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
            elif si and not sj:
                cnt += 1
            elif not si and sj:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # 1. Precompute comparability bitsets
    # comp_bitset[i] is a bitmask where bit j is 1 if mask i and mask j are comparable
    comp_bitset = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j and ((i & j) == i or (i & j) == j):
                ci |= (1 << j)
        comp_bitset[i] = ci

    # 2. Greedy Initialization
    # Start with a random set of masks or a specific heuristic
    # Heuristic: Pick masks that are comparable to many others.
    # A mask of size 6 is comparable to $\binom{12}{k}$ for k<6 and k>6? No.
    # Size k mask is comparable to all masks of size <k and >k that contain/are contained.
    # Actually, let's just use a simple greedy: pick the mask with highest degree in the comparability graph,
    # then remove its neighbors? No, we want high comparability.
    # Greedy: pick mask that maximizes comparability with already picked.
    
    # Better Init: Random restarts with local search.
    
    best_score_global = -1
    best_selected_global = []
    
    # Run multiple restarts to avoid local optima
    # Time budget for restarts: use most of the time.
    # Each restart: Greedy + Local Search.
    
    num_restarts = 100 # We will break early if time runs out
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize selected set
        # Strategy: Pick m random masks? Or greedy?
        # Greedy is better.
        selected = []
        selected_mask = 0
        available = list(range(total_masks))
        random.shuffle(available)
        
        # To make greedy faster and more effective, we can sort available by their total degree?
        # Or just pick the best from available at each step.
        # Picking best from 4096 each step is 40 * 4096 ops.
        
        for _ in range(m):
            best_mask = -1
            best_score = -1
            # We can iterate through all available, but checking 'in selected' is slow.
            # Using selected_mask bit check is fast.
            for k in available:
                if (selected_mask >> k) & 1:
                    continue
                score = (comp_bitset[k] & selected_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = k
            
            if best_mask == -1:
                # Should not happen if m <= total_masks
                break
                
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
            
        # Calculate initial score
        current_score = 0
        for i in range(m):
            current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
        current_score //= 2
        
        if current_score > best_score_global:
            best_score_global = current_score
            best_selected_global = list(selected)
            
        # Local Search: 1-Swaps
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle the order of indices to break symmetry?
            # Or just iterate.
            for idx in range(m):
                if time.time() > deadline - 2.0:
                    break
                    
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_gain = -1
                best_delta = 0
                
                # Find best new mask
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_gain = gain
                        best_new = new_mask
                
                if best_delta > 0 and best_new != -1:
                    # Perform swap
                    selected[idx] = best_new
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new)
                    current_score += best_delta
                    improved = True
                    
                    if current_score > best_score_global:
                        best_score_global = current_score
                        best_selected_global = list(selected)
                    
                    # Optional: Break out of idx loop to restart scanning?
                    # Or continue? Continuing is fine, we might find more.
                    # But restarting ensures we don't miss interactions.
                    # Let's continue for now, it's faster.
        
        # Local Search: 2-Swaps (Limited)
        # If time permits, try 2-swaps to escape local optima.
        # We will sample a few 2-swaps.
        if time.time() < deadline - 1.0:
            # Try 200 random 2-swaps
            for _ in range(200):
                if time.time() > deadline - 1.0:
                    break
                
                # Pick two random indices to remove
                idx1, idx2 = random.sample(range(m), 2)
                old1 = selected[idx1]
                old2 = selected[idx2]
                
                sel_without_old12 = selected_mask & ~(1 << old1) & ~(1 << old2)
                
                loss = (comp_bitset[old1] & sel_without_old12).bit_count() + \
                       (comp_bitset[old2] & sel_without_old12).bit_count()
                
                # Find best pair of new masks
                # This is expensive. We will approximate by picking best single mask,
                # then best second mask.
                
                # Find best new1
                best_new1 = -1
                best_gain1 = -1
                for n1 in range(total_masks):
                    if (selected_mask >> n1) & 1:
                        continue
                    g1 = (comp_bitset[n1] & sel_without_old12).bit_count()
                    if g1 > best_gain1:
                        best_gain1 = g1
                        best_new1 = n1
                        
                if best_new1 == -1:
                    continue
                    
                # Temporarily add new1 to calculate best new2
                sel_with_new1 = sel_without_old12 | (1 << best_new1)
                
                best_new2 = -1
                best_gain2 = -1
                for n2 in range(total_masks):
                    if (selected_mask >> n2) & 1:
                        continue
                    if n2 == best_new1:
                        continue
                    g2 = (comp_bitset[n2] & sel_with_new1).bit_count()
                    # Note: g2 includes comparability with new1, which is good.
                    if g2 > best_gain2:
                        best_gain2 = g2
                        best_new2 = n2
                        
                if best_new2 == -1:
                    continue
                
                gain = best_gain1 + best_gain2
                delta = gain - loss
                
                if delta > 0:
                    selected[idx1] = best_new1
                    selected[idx2] = best_new2
                    selected_mask = (selected_mask & ~(1 << old1) & ~(1 << old2)) | (1 << best_new1) | (1 << best_new2)
                    current_score += delta
                    
                    if current_score > best_score_global:
                        best_score_global = current_score
                        best_selected_global = list(selected)

    # Convert best_selected_global to lists of integers
    result = []
    for mask in best_selected_global:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
        
    return final_result
