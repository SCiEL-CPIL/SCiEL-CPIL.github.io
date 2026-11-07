# EVOLVE-BLOCK-START
import time
import random

def _compute_cardinality_sums_bitset(elements):
    """Compute bitsets of achievable sums for each cardinality."""
    n = len(elements)
    if n == 0:
        return [1]  # sums[0] = 1 (bit 0 set)
    
    # sums[k] is an integer where bit s is 1 if sum s is achievable with exactly k elements
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 is achievable with 0 elements
    
    for elem in elements:
        # Update in reverse to avoid using elem multiple times
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << elem)
            
    return sums

def _is_valid(elements):
    """Check if the set satisfies the cardinality-separated subset sum constraint using bitsets."""
    sums = _compute_cardinality_sums_bitset(elements)
    n = len(elements)
    # Check for intersection between any two different cardinality classes
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
    return True

def _check_add(elements, new_elem):
    """Check if adding new_elem to elements maintains validity."""
    if new_elem in elements:
        return False
    new_elements = list(elements)
    new_elements.append(new_elem)
    new_elements.sort()
    return _is_valid(new_elements)

def _grow_and_search(current, candidates, deadline):
    """
    Iteratively try to grow the set by:
    1. Adding a single element.
    2. If blocked, trying 1-for-2 swaps (remove 1, add 2) to increase size.
    3. If blocked, trying 1-for-1 swaps to rearrange, then retrying growth.
    """
    current = sorted(current)
    used = set(current)
    
    # Pre-filter candidates to those not in current
    available = [c for c in candidates if c not in used]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add a single element
        added = False
        for cand in available:
            if time.time() > deadline:
                break
            if _check_add(current, cand):
                current.append(cand)
                current.sort()
                used.add(cand)
                available.remove(cand)
                improved = True
                added = True
                break
        
        if added:
            continue # Restart loop to try adding more
            
        # Phase 2: If no single add works, try 1-for-2 swaps (increase size by 1)
        # Remove one element 'rem', add two 'a', 'b'
        n = len(current)
        if n > 0:
            for i in range(n):
                if time.time() > deadline:
                    break
                rem = current[i]
                # Try pairs from available
                # Optimization: limit pairs to avoid O(N^3) explosion
                # We can just iterate through available and pick pairs
                avail_list = available
                for j in range(len(avail_list)):
                    if time.time() > deadline:
                        break
                    a = avail_list[j]
                    for k in range(j+1, len(avail_list)):
                        if time.time() > deadline:
                            break
                        b = avail_list[k]
                        
                        new_set = current[:i] + current[i+1:] + [a, b]
                        new_set.sort()
                        if _is_valid(new_set):
                            current = new_set
                            used.discard(rem)
                            used.add(a)
                            used.add(b)
                            # Update available
                            if a in available: available.remove(a)
                            if b in available: available.remove(b)
                            if rem not in used: available.append(rem)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if improved:
            continue # Restart loop
            
        # Phase 3: If no growth, try 1-for-1 swaps to rearrange
        swapped = False
        for i in range(n):
            if time.time() > deadline:
                break
            rem = current[i]
            for cand in available:
                if time.time() > deadline:
                    break
                new_set = current[:i] + current[i+1:] + [cand]
                new_set.sort()
                if _is_valid(new_set):
                    current = new_set
                    used.discard(rem)
                    used.add(cand)
                    # Update available
                    available.remove(cand)
                    available.append(rem)
                    swapped = True
                    break
            if swapped:
                break
        
        if swapped:
            improved = True # Allow loop to continue to try growing again
            # To prevent infinite loops on swaps, we could track state, 
            # but the "add" phase usually resolves it or we timeout.
            # We'll rely on the fact that successful adds increase size, 
            # and swaps are only tried if adds fail.
            # However, if swaps cycle, we might loop. 
            # Simple heuristic: limit consecutive swaps without an add?
            # For now, we just let it run until deadline.
            
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s budget, 5s safety margin
    
    # Base solution: Large numbers are generally easier to separate
    # Try a few different base strategies
    best = []
    
    # Strategy 1: Start with top 30
    base1 = list(range(227, 257)) # 30 elements
    candidates1 = [x for x in range(1, 257) if x not in set(base1)]
    random.shuffle(candidates1)
    res1 = _grow_and_search(base1, candidates1, deadline)
    if len(res1) > len(best):
        best = res1
        
    if time.time() > deadline:
        return best
        
    # Strategy 2: Start with a random subset of large numbers
    # Maybe 20 random from top 50
    top_50 = list(range(207, 257))
    random.shuffle(top_50)
    base2 = top_50[:20]
    candidates2 = [x for x in range(1, 257) if x not in set(base2)]
    random.shuffle(candidates2)
    res2 = _grow_and_search(base2, candidates2, deadline)
    if len(res2) > len(best):
        best = res2
        
    if time.time() > deadline:
        return best
        
    # Strategy 3: Greedy from scratch (smallest to largest or largest to smallest)
    # Largest to smallest greedy
    current3 = []
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if _check_add(current3, x):
            current3.append(x)
            
    if len(current3) > len(best):
        best = current3
        
    # Refine the best found so far
    if time.time() < deadline:
        best_set = set(best)
        candidates_best = [x for x in range(1, 257) if x not in best_set]
        random.shuffle(candidates_best)
        res_final = _grow_and_search(best, candidates_best, deadline)
        if len(res_final) > len(best):
            best = res_final
            
    return best
# EVOLVE-BLOCK-END
