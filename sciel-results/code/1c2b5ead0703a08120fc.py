"""Rewritten program for the suite task; maximizes comparable pairs in a 40-subset family on 12 vertices."""
import itertools
import time

# EVOLVE-BLOCK-START

def _is_strict_subset(a, b):
    """Return True if a is a strict subset of b (both are int bitmasks)."""
    return (a & b) == a and a != b

def _count_comparable(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    cnt = 0
    n = len(masks)
    for i in range(n):
        ai = masks[i]
        for j in range(i + 1, n):
            aj = masks[j]
            if _is_strict_subset(ai, aj) or _is_strict_subset(aj, ai):
                cnt += 1
    return cnt

def _greedy_build(n, m, time_limit=None):
    """
    Greedy construction: iteratively add the mask that maximizes the number of
    new comparable pairs with the already-chosen masks.
    """
    if time_limit is None:
        time_limit = 50.0
    start = time.time()
    
    all_masks = list(range(1 << n))
    chosen = []
    chosen_set = set()
    
    # Precompute subset relation? For 4096 masks, we can precompute for each mask
    # the list of masks it is a subset of (or comparable to). But that's 4096*4096 = 16M entries.
    # Instead, compute on the fly.
    
    for step in range(m):
        if time.time() - start > time_limit - 5:
            break
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in chosen_set:
                continue
            # Score = number of comparable pairs with existing chosen masks
            score = 0
            for cm in chosen:
                if _is_strict_subset(mask, cm) or _is_strict_subset(cm, mask):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            break
        chosen.append(best_mask)
        chosen_set.add(best_mask)
    
    return chosen

def _local_search(masks, time_limit=20.0):
    """
    Local search: try swapping one mask with another to improve the count.
    """
    start = time.time()
    n = 12
    all_masks = list(range(1 << n))
    current_score = _count_comparable(masks)
    best_masks = list(masks)
    best_score = current_score
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(len(best_masks)):
            if time.time() - start > time_limit:
                break
            old_mask = best_masks[i]
            for new_mask in all_masks:
                if new_mask in best_masks:
                    continue
                # Compute new score by removing old_mask and adding new_mask
                # Score change = (comparables of new_mask with others) - (comparables of old_mask with others)
                # where "others" excludes the mask itself
                gain = 0
                loss = 0
                for j in range(len(best_masks)):
                    if j == i:
                        continue
                    mj = best_masks[j]
                    if _is_strict_subset(new_mask, mj) or _is_strict_subset(mj, new_mask):
                        gain += 1
                    if _is_strict_subset(old_mask, mj) or _is_strict_subset(mj, old_mask):
                        loss += 1
                delta = gain - loss
                if delta > 0:
                    best_masks[i] = new_mask
                    best_score += delta
                    improved = True
                    break  # restart inner loop with new best_masks
            if improved:
                break
    
    return best_masks, best_score

def _simulated_annealing(masks, time_limit=30.0):
    """
    Simulated annealing to escape local optima.
    """
    import random
    start = time.time()
    n = 12
    all_masks = list(range(1 << n))
    m = len(masks)
    
    current = list(masks)
    current_score = _count_comparable(current)
    best = list(current)
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while T > T_min and (time.time() - start) < time_limit:
        # Pick a random position and a random replacement
        i = random.randint(0, m - 1)
        old_mask = current[i]
        new_mask = random.choice(all_masks)
        if new_mask in current:
            T *= cooling
            continue
        
        # Compute score change
        gain = 0
        loss = 0
        for j in range(m):
            if j == i:
                continue
            mj = current[j]
            if _is_strict_subset(new_mask, mj) or _is_strict_subset(mj, new_mask):
                gain += 1
            if _is_strict_subset(old_mask, mj) or _is_strict_subset(mj, old_mask):
                loss += 1
        delta = gain - loss
        
        if delta > 0:
            current[i] = new_mask
            current_score += delta
            if current_score > best_score:
                best = list(current)
                best_score = current_score
        else:
            import math
            if random.random() < math.exp(delta / T):
                current[i] = new_mask
                current_score += delta
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 12
    m = 40
    
    # Phase 1: Greedy construction
    greedy_result = _greedy_build(n, m, time_limit=10.0)
    if len(greedy_result) < m:
        # Fill remaining with random masks not already chosen
        chosen_set = set(greedy_result)
        for mask in range(1 << n):
            if len(greedy_result) >= m:
                break
            if mask not in chosen_set:
                greedy_result.append(mask)
                chosen_set.add(mask)
    
    current_best = list(greedy_result)
    current_score = _count_comparable(current_best)
    
    # Phase 2: Local search
    if time.time() < deadline - 15.0:
        ls_result, ls_score = _local_search(current_best, time_limit=10.0)
        if ls_score > current_score:
            current_best = ls_result
            current_score = ls_score
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_result, sa_score = _simulated_annealing(current_best, time_limit=15.0)
        if sa_score > current_score:
            current_best = sa_result
            current_score = sa_score
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 1.0:
        ls_result, ls_score = _local_search(current_best, time_limit=5.0)
        if ls_score > current_score:
            current_best = ls_result
            current_score = ls_score
    
    # Convert to list of lists
    result = [[b for b in range(n) if (mask >> b) & 1] for mask in current_best]
    
    # Ensure exactly 40 distinct lists
    if len(result) != m:
        # Pad or trim
        seen = set()
        final = []
        for r in result:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                final.append(r)
        while len(final) < m:
            for mask in range(1 << n):
                t = tuple([b for b in range(n) if (mask >> b) & 1])
                if t not in seen:
                    seen.add(t)
                    final.append([b for b in range(n) if (mask >> b) & 1])
                    break
            else:
                break
        result = final[:m]
    
    return result

# EVOLVE-BLOCK-END
