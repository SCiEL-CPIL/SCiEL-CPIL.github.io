import time
import random

def _subset_sums_bitset(distances):
    bits = 1
    for d in distances:
        bits |= (bits << d)
    return bits

def _is_valid(selected):
    n = len(selected)
    if n < 3:
        return True
    selected_sorted = sorted(selected)
    for i, a in enumerate(selected_sorted):
        # Distances to smaller elements
        # x < a => a - x > 0
        D_less = [a - x for x in selected_sorted[:i]]
        # Distances to larger elements
        # y > a => y - a > 0
        D_greater = [x - a for x in selected_sorted[i+1:]]
        
        sums_less = _subset_sums_bitset(D_less)
        sums_greater = _subset_sums_bitset(D_greater)
        
        # Check for common positive sum
        common = sums_less & sums_greater
        if common & ~1:
            return False
    return True

def _greedy_construct(order):
    selected = []
    for c in order:
        candidate = selected + [c]
        if _is_valid(candidate):
            selected = candidate
    return selected

def _try_fill(selected):
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        # Iterate in a somewhat random order or just 1..512
        # 1..512 is fine, it's fast
        for c in range(1, 513):
            if c not in selected_set:
                candidate = list(selected_set) + [c]
                if _is_valid(candidate):
                    selected_set.add(c)
                    changed = True
    return list(selected_set)

def construct():
    start_time = time.time()
    deadline = 80  # Safety margin
    
    best_set = []
    
    # 1. Descending
    s = _greedy_construct(range(512, 0, -1))
    s = _try_fill(s)
    if len(s) > len(best_set):
        best_set = s
        
    # 2. Ascending
    s = _greedy_construct(range(1, 513))
    s = _try_fill(s)
    if len(s) > len(best_set):
        best_set = s
        
    # 3. Randomized Greedy
    # We can run many iterations
    iterations = 0
    while time.time() - start_time < deadline:
        iterations += 1
        random_order = list(range(1, 513))
        random.shuffle(random_order)
        s = _greedy_construct(random_order)
        s = _try_fill(s)
        if len(s) > len(best_set):
            best_set = s
        # If we found 48, we can stop early? 
        # The task says "Maximize cardinality. At most 48".
        # If we hit 48, that's the max possible.
        if len(best_set) == 48:
            break
            
    return best_set
