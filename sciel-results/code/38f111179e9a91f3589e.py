import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    """
    n = P['n']
    m = len(sets)
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                if mi != mj:
                    count += 1
    return count

def _random_initial_solution():
    n = P['n']
    m = P['m']
    all_masks = [0] * (1 << n)
    chosen = set()
    while len(chosen) < m:
        mask = random.randint(0, (1 << n) - 1)
        chosen.add(mask)
    return list(chosen)

def _local_search(initial_masks, deadline):
    n = P['n']
    m = P['m']
    current_masks = list(initial_masks)
    
    # Precompute comp_bitset for all 4096 masks
    total_masks = 1 << n
    all_ones = total_masks - 1
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        if i != 0:
            sub = (i - 1) & i
            while True:
                ci |= (1 << sub)
                if sub == 0:
                    break
                sub = (sub - 1) & i
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    # Initialize selected set and score
    selected = list(current_masks)
    selected_mask = 0
    for mask in selected:
        selected_mask |= (1 << mask)
    
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local search: single swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Try to find a better mask
            best_new_mask = -1
            best_gain = -1
            
            # Random sampling to speed up search
            for _ in range(200):
                new_mask = random.randint(0, total_masks - 1)
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_new_mask = new_mask
            
            if best_new_mask != -1 and best_gain > 0:
                selected[idx] = best_new_mask
                selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
                improved = True
                break  # restart scan
    
    return best_score, best_selected

def construct():
    deadline = time.time() + 50.0 # 50s safety margin
    
    best_overall_score = -1
    best_overall_masks = None
    
    # Multiple restarts
    while time.time() < deadline:
        init_masks = _random_initial_solution()
        score, masks = _local_search(init_masks, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_masks = masks
        
        # If we hit a very high score, we might break early? 
        # Max possible pairs is C(40,2) = 780. 
        # A chain of 13 has C(13,2)=78. 
        # If we have multiple chains, say 3 chains of 13, 13, 14? 
        # C(13,2)+C(13,2)+C(14,2) = 78+78+91 = 247.
        # If we can get close to 780, that's great.
        # But with n=12, we can't have a chain of 40.
        # Max chain length is 13.
        # So we are limited.
        # Let's just run until deadline.
        
        # If the score is very high, e.g., > 400, maybe stop? 
        # No, keep searching for better.
        
    # Convert best_overall_masks to lists of integers
    n = P['n']
    result = []
    for mask in best_overall_masks:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
