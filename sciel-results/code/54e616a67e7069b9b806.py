# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # Safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    all_ones = total_masks - 1  # 0xFFF
    
    # Precompute comparability bitsets
    # comp_bitset[i] has bit j set iff i and j are strictly comparable
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Proper submasks of i
        sub = i
        while sub:
            ci |= (1 << sub)
            sub = (sub - 1) & i
        # Proper supermasks of i
        comp = all_ones ^ i
        if comp:
            sub = comp
            while sub:
                s = i | sub
                ci |= (1 << s)
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    def count_pairs(selected_mask, selected_list):
        """Count comparable pairs in the selected set."""
        total = 0
        for i in range(m):
            total += (comp_bitset[selected_list[i]] & selected_mask).bit_count()
        return total // 2
    
    def greedy_initial():
        """Greedy: start with empty set, then add masks maximizing comparability."""
        selected = []
        selected_mask = 0
        
        # Start with empty set (comparable with everything)
        selected.append(0)
        selected_mask |= 1
        
        for _ in range(m - 1):
            best_mask = -1
            best_score = -1
            for k in range(total_masks):
                if (selected_mask >> k) & 1:
                    continue
                score = (comp_bitset[k] & selected_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = k
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
        
        return selected, selected_mask, count_pairs(selected_mask, selected)
    
    def chain_initial():
        """Initialize with a long chain, then fill greedily."""
        selected = []
        selected_mask = 0
        
        # Build a chain: empty, {0}, {0,1}, {0,1,2}, ... up to size 12
        mask = 0
        for i in range(n):
            selected.append(mask)
            selected_mask |= (1 << mask)
            mask |= (1 << i)
        # Now mask is all_ones, add it
        selected.append(mask)
        selected_mask |= (1 << mask)
        # That's 13 sets in the chain
        
        # Fill remaining greedily
        for _ in range(m - len(selected)):
            best_mask = -1
            best_score = -1
            for k in range(total_masks):
                if (selected_mask >> k) & 1:
                    continue
                score = (comp_bitset[k] & selected_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = k
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
        
        return selected, selected_mask, count_pairs(selected_mask, selected)
    
    def local_search(selected, selected_mask, current_score, time_limit):
        """Exhaustive single-swap local search with best-improvement per position."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx in range(m):
                if time.time() >= time_limit:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_mask
                
                if best_new >= 0:
                    selected[idx] = best_new
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new)
                    current_score += best_delta
                    improved = True
        
        return selected, selected_mask, current_score
    
    # Try multiple initializations and keep the best
    best_score = -1
    best_selected = None
    best_mask = 0
    
    # Initial strategy 1: greedy from empty
    sel1, mask1, score1 = greedy_initial()
    if score1 > best_score:
        best_score = score1
        best_selected = list(sel1)
        best_mask = mask1
    
    # Initial strategy 2: chain-based
    sel2, mask2, score2 = chain_initial()
    if score2 > best_score:
        best_score = score2
        best_selected = list(sel2)
        best_mask = mask2
    
    # Local search on the best initial
    sel, smask, score = local_search(list(best_selected), best_mask, best_score, deadline - 5.0)
    if score > best_score:
        best_score = score
        best_selected = list(sel)
        best_mask = smask
    
    # If we have time, try local search on the other initial too
    if time.time() < deadline - 5.0:
        other_sel, other_mask, other_score = (sel2, mask2, score2) if best_mask == mask1 else (sel1, mask1, score1)
        sel3, smask3, score3 = local_search(list(other_sel), other_mask, other_score, deadline - 3.0)
        if score3 > best_score:
            best_score = score3
            best_selected = list(sel3)
            best_mask = smask3
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
