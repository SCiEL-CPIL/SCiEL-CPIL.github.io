import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Group masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        masks_by_size[popcount[mask]].append(mask)
        
    # Helper: check if a is strict subset of b or vice versa
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # --- Phase 1: Structural Initialization ---
    # We want to pick sets that are highly connected in the comparability graph.
    # 1. Empty set and Full set are comparable with everything.
    # 2. Singletons are subsets of many sets.
    # 3. Co-singletons are supersets of many sets.
    
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}
    
    # Add all singletons
    for i in range(n):
        mask = 1 << i
        current_masks.append(mask)
        used_masks.add(mask)
        
    # Add all co-singletons
    for i in range(n):
        mask = full_mask ^ (1 << i)
        current_masks.append(mask)
        used_masks.add(mask)
        
    # Current size: 1 + 1 + 12 + 12 = 26. Need 14 more.
    # Greedily add the next best set: the one that is comparable with the most existing sets.
    
    # To speed up greedy, we can just iterate through all unused masks and count comparables.
    # There are 4096 - 26 = 4070 unused masks.
    # For each, check against 26 existing masks. 4070 * 26 ~ 100k ops. Very fast.
    # Repeat 14 times.
    
    all_masks = list(range(1 << n))
    # Shuffle to break ties deterministically but randomly
    random.seed(42)
    random.shuffle(all_masks)
    
    for _ in range(m - len(current_masks)):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in used_masks:
                continue
            score = 0
            for cm in current_masks:
                if is_comparable(mask, cm):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            # Fallback if something went wrong (should not happen)
            for mask in all_masks:
                if mask not in used_masks:
                    current_masks.append(mask)
                    used_masks.add(mask)
                    break

    # --- Phase 2: Local Search ---
    # We maintain an array of the current masks.
    # We maintain an array `comp_counts` where comp_counts[i] is the number of other masks in current_masks
    # that are comparable with current_masks[i].
    
    def build_comp_counts(masks):
        counts = [0] * len(masks)
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    counts[i] += 1
                    counts[j] += 1
        return counts

    comp_counts = build_comp_counts(current_masks)
    current_score = sum(comp_counts) // 2
    
    best_score = current_score
    best_solution = list(current_masks)
    
    # Precompute comparability with all other masks? No, too big.
    # Instead, for a candidate swap (remove u, add v):
    # Delta = (comp(v, S\{u}) - comp(u, S\{u}))
    # comp(u, S\{u}) is exactly comp_counts[u_idx].
    # comp(v, S\{u}) requires checking v against all 39 other masks.
    
    # Optimization: 
    # We can precompute for each mask v, its comparability with each mask in S? 
    # That's 4096 * 40 booleans. 160k booleans. ~20KB. Very fast to compute and update.
    
    # Let's maintain a matrix `comp_with_current` where comp_with_current[v][i] is True if mask v is comparable with current_masks[i].
    # This allows O(1) check for "is v comparable with current_masks[i]".
    # Then comp(v, S\{u}) is sum(comp_with_current[v][j] for j != u_idx).
    
    comp_with_current = [[False] * m for _ in range(1 << n)]
    for i in range(m):
        mi = current_masks[i]
        for v in range(1 << n):
            if v != mi: # Self is not strict subset
                comp_with_current[v][i] = is_comparable(v, mi)
                
    # To make "sum" faster, we can precompute the count for each v?
    # count_with_current[v] = number of i such that comp_with_current[v][i] is True.
    # Then comp(v, S\{u}) = count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0).
    
    count_with_current = [0] * (1 << n)
    for v in range(1 << n):
        count_with_current[v] = sum(comp_with_current[v])
        
    # Now the delta calculation is O(1).
    # Delta = (count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)) - comp_counts[u_idx]
    
    # We need to update comp_counts and count_with_current when we accept a swap.
    # If we swap u (at index u_idx) with v:
    # 1. For all j != u_idx:
    #    if comp_with_current[u][j]: comp_counts[j] -= 1
    #    if comp_with_current[v][j]: comp_counts[j] += 1
    # 2. Update comp_counts[u_idx] to be the new count for v? No, v takes the slot.
    #    The new count for the element at u_idx (which is now v) is count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)?
    #    Wait, comp_counts[i] is the number of *other* masks comparable with masks[i].
    #    So new_comp_counts[u_idx] = count_with_current[v] - 1 (since v is not comparable with itself, but count_with_current includes self? No, we excluded self in init).
    #    Actually, count_with_current[v] counts comparables in the *current* set S.
    #    When we remove u and add v, the set becomes S' = S \ {u} U {v}.
    #    The comparables of v in S' are the comparables of v in S\{u}.
    #    So new_comp_counts[u_idx] = count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0).
    #    This is correct.
    
    # 3. Update comp_with_current for the new mask v?
    #    We already have comp_with_current for all v.
    #    But we need to update comp_with_current for all other masks w?
    #    No, comp_with_current[v][i] is "is v comparable with current_masks[i]".
    #    When current_masks[u_idx] changes from u to v:
    #    - For any other mask w (not u, not v), comp_with_current[w][u_idx] should now reflect comparability with v.
    #    - So we need to update column u_idx of comp_with_current for all w.
    #    - That is O(4096) per swap.
    #    - 4096 ops is small.
    
    # 4. Update count_with_current for all w?
    #    count_with_current[w] is the sum of row w.
    #    If column u_idx changes, count_with_current[w] changes by (new_val - old_val).
    #    So we can update count_with_current[w] in O(1) per w.
    #    Total update cost per swap: O(4096).
    
    # Total cost per iteration: O(1) to find best swap? No, we need to find the best v.
    # We can iterate through all unused v (4096 - 40) and calculate delta in O(1).
    # So finding best swap is O(4056).
    # Updating is O(4096).
    # Total per iteration ~ 8000 ops.
    # In 50 seconds, we can do millions of iterations.
    
    # Let's implement this.
    
    # Initial best
    best_delta = 0
    best_u_idx = -1
    best_v = -1
    
    # To avoid O(N^2) search for best swap every time, we can just do random swaps or hill climbing.
    # Hill climbing: find the best swap.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if time.time() > deadline - 2.0:
            break
            
        # Find best swap
        max_delta = 0
        best_u_idx_candidate = -1
        best_v_candidate = -1
        
        # Iterate through all u in current set
        for u_idx in range(m):
            u_mask = current_masks[u_idx]
            u_comp = comp_counts[u_idx]
            
            # Iterate through all v not in current set
            # Optimization: only check v that are "likely" to be good?
            # For now, check all.
            for v in range(1 << n):
                if v in used_masks:
                    continue
                # Delta = (count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)) - u_comp
                v_comp_in_s_minus_u = count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)
                delta = v_comp_in_s_minus_u - u_comp
                if delta > max_delta:
                    max_delta = delta
                    best_u_idx_candidate = u_idx
                    best_v_candidate = v
            
        if max_delta <= 0:
            # No improvement found. 
            # We could try simulated annealing to escape local minima.
            # Or just stop.
            # Given the structure, hill climbing usually gets close to optimal.
            # Let's add a small random perturbation if stuck?
            # For simplicity and speed, let's just break or do random swaps.
            # Let's try random swaps for a bit to escape local optima.
            if iterations % 100 == 0:
                # Random swap
                u_idx = random.randint(0, m - 1)
                # Pick random unused v
                unused = [v for v in range(1 << n) if v not in used_masks]
                if not unused:
                    break
                v = random.choice(unused)
                
                # Calculate delta
                u_mask = current_masks[u_idx]
                u_comp = comp_counts[u_idx]
                v_comp_in_s_minus_u = count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)
                delta = v_comp_in_s_minus_u - u_comp
                
                # Accept with probability based on delta (Simulated Annealing)
                # T = 5.0
                if delta >= 0 or random.random() < math.exp(delta / 5.0):
                    # Perform swap
                    old_u = current_masks[u_idx]
                    current_masks[u_idx] = v
                    used_masks.remove(old_u)
                    used_masks.add(v)
                    
                    # Update comp_counts
                    for j in range(m):
                        if j == u_idx:
                            continue
                        if comp_with_current[old_u][j]:
                            comp_counts[j] -= 1
                        if comp_with_current[v][j]:
                            comp_counts[j] += 1
                    comp_counts[u_idx] = v_comp_in_s_minus_u
                    
                    # Update comp_with_current column u_idx and count_with_current
                    for w in range(1 << n):
                        old_val = comp_with_current[w][u_idx]
                        new_val = is_comparable(w, v)
                        if old_val != new_val:
                            comp_with_current[w][u_idx] = new_val
                            if new_val:
                                count_with_current[w] += 1
                            else:
                                count_with_current[w] -= 1
                                
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = list(current_masks)
            else:
                # If we are stuck in hill climbing, break
                break
        else:
            # Perform the best swap
            u_idx = best_u_idx_candidate
            v = best_v_candidate
            old_u = current_masks[u_idx]
            
            current_masks[u_idx] = v
            used_masks.remove(old_u)
            used_masks.add(v)
            
            # Update comp_counts
            for j in range(m):
                if j == u_idx:
                    continue
                if comp_with_current[old_u][j]:
                    comp_counts[j] -= 1
                if comp_with_current[v][j]:
                    comp_counts[j] += 1
            comp_counts[u_idx] = count_with_current[v] - (1 if comp_with_current[v][u_idx] else 0)
            
            # Update comp_with_current column u_idx and count_with_current
            for w in range(1 << n):
                old_val = comp_with_current[w][u_idx]
                new_val = is_comparable(w, v)
                if old_val != new_val:
                    comp_with_current[w][u_idx] = new_val
                    if new_val:
                        count_with_current[w] += 1
                    else:
                        count_with_current[w] -= 1
                        
            current_score += max_delta
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_masks)

    # Convert best_solution to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
