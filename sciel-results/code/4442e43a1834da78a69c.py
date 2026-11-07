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

def _greedy_initial_solution(comp_bitset, deadline):
    """Build a strong initial solution by greedily adding the mask with highest compatibility."""
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    selected = []
    selected_bool = [False] * total_masks
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in range(total_masks):
            if selected_bool[mask]:
                continue
            # Count how many already-selected sets are comparable with mask
            comp_count = 0
            for sel in selected:
                if (comp_bitset[mask] >> sel) & 1:
                    comp_count += 1
            if comp_count > best_score:
                best_score = comp_count
                best_mask = mask
        if best_mask == -1:
            break
        selected.append(best_mask)
        selected_bool[best_mask] = True
    
    # Fill remaining with random if needed
    while len(selected) < m:
        mask = random.randint(0, total_masks - 1)
        if not selected_bool[mask]:
            selected.append(mask)
            selected_bool[mask] = True
    
    return selected

def _local_search(initial_masks, comp_bitset, deadline):
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    selected = list(initial_masks)
    selected_bool = [False] * total_masks
    for mask in selected:
        selected_bool[mask] = True
    
    # Compute current score
    current_score = 0
    for i in range(m):
        for j in range(i + 1, m):
            if (comp_bitset[selected[i]] >> selected[j]) & 1:
                current_score += 1
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local search: single swaps
    improved = True
    no_improve_count = 0
    max_no_improve = 10
    
    while improved and time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            
            # Compute loss: how many other selected sets are comparable with old_mask
            loss = 0
            for j in range(m):
                if j == idx:
                    continue
                if (comp_bitset[old_mask] >> selected[j]) & 1:
                    loss += 1
            
            # Find best replacement
            best_new_mask = -1
            best_gain = -1
            
            for new_mask in range(total_masks):
                if selected_bool[new_mask]:
                    continue
                gain = 0
                for j in range(m):
                    if j == idx:
                        continue
                    if (comp_bitset[new_mask] >> selected[j]) & 1:
                        gain += 1
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_new_mask = new_mask
            
            if best_new_mask != -1 and best_gain > 0:
                # Perform the swap
                selected_bool[old_mask] = False
                selected_bool[best_new_mask] = True
                selected[idx] = best_new_mask
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
                improved = True
                no_improve_count = 0
                break  # Restart scan after a move
        
        if not improved:
            no_improve_count += 1
            # Try a random perturbation to escape local optimum
            if no_improve_count >= 3 and time.time() < deadline:
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, m - 1)
                    old_mask = selected[idx]
                    new_mask = random.randint(0, total_masks - 1)
                    while selected_bool[new_mask]:
                        new_mask = random.randint(0, total_masks - 1)
                    
                    # Compute delta
                    loss = 0
                    gain = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        if (comp_bitset[old_mask] >> selected[j]) & 1:
                            loss += 1
                        if (comp_bitset[new_mask] >> selected[j]) & 1:
                            gain += 1
                    delta = gain - loss
                    
                    # Accept with some probability if delta is negative (simulated annealing style)
                    if delta >= 0 or random.random() < 0.3:
                        selected_bool[old_mask] = False
                        selected_bool[new_mask] = True
                        selected[idx] = new_mask
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
    
    return best_score, best_selected

def construct():
    deadline = time.time() + 50.0  # 50s safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    all_ones = total_masks - 1
    
    # Precompute comp_bitset for all 4096 masks
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
    
    best_overall_score = -1
    best_overall_masks = None
    
    # Greedy initialization
    greedy_masks = _greedy_initial_solution(comp_bitset, deadline)
    score, masks = _local_search(greedy_masks, comp_bitset, deadline)
    if score > best_overall_score:
        best_overall_score = score
        best_overall_masks = masks
    
    # Multiple random restarts
    while time.time() < deadline:
        # Random initial solution with bias toward chains
        chosen = set()
        # Start with some structured sets
        chosen.add(0)  # empty set
        chosen.add(all_ones)  # full set
        
        # Add sets of various sizes to encourage comparability
        for size in [1, 2, 3, 6, 9, 10, 11]:
            subset = random.sample(range(n), size)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            chosen.add(mask)
        
        while len(chosen) < m:
            mask = random.randint(0, total_masks - 1)
            chosen.add(mask)
        init_masks = list(chosen)
        
        score, masks = _local_search(init_masks, comp_bitset, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_masks = masks
    
    # Convert best_overall_masks to lists of integers
    result = []
    for mask in best_overall_masks:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
