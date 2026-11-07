import time
import random

def _try_add(arr, v, sum_to_card, current_max_sum):
    """Try adding v to arr. Returns (success, new_sum_to_card, new_max_sum) or (False, None, None)."""
    max_possible_sum = 60 * 256
    n = len(arr)
    if n >= 60:
        return False, None, None
    
    # Check if adding v causes conflicts
    valid = True
    new_entries = []
    for s in range(current_max_sum + 1):
        if sum_to_card[s] >= 0:
            ns = s + v
            nk = sum_to_card[s] + 1
            if sum_to_card[ns] >= 0 and sum_to_card[ns] != nk:
                valid = False
                break
            new_entries.append((ns, nk))
    
    if not valid:
        return False, None, None
    
    # Apply
    new_sum_to_card = sum_to_card[:]
    for ns, nk in new_entries:
        if new_sum_to_card[ns] < 0 or new_sum_to_card[ns] == nk:
            new_sum_to_card[ns] = nk
    
    return True, new_sum_to_card, current_max_sum + v

def _grow_greedy(ordering, deadline):
    """Grow a set greedily using the given ordering."""
    current_set = []
    max_possible_sum = 60 * 256
    sum_to_card = [-1] * (max_possible_sum + 1)
    sum_to_card[0] = 0
    current_max_sum = 0
    
    for x in ordering:
        if len(current_set) >= 60:
            break
        if time.time() > deadline:
            break
        
        success, new_s2c, new_max = _try_add(current_set, x, sum_to_card, current_max_sum)
        if success:
            current_set.append(x)
            sum_to_card = new_s2c
            current_max_sum = new_max
    
    return current_set

def _try_swap(arr, sum_to_card, current_max_sum, i, new_val):
    """Try replacing arr[i] with new_val. Returns (success, new_sum_to_card, new_max_sum) or (False, None, None)."""
    if new_val in arr:
        return False, None, None
    
    # Remove arr[i] and try to add new_val
    # First, we need to remove arr[i] from the DP state
    # This is complex, so we rebuild from scratch for the remaining elements
    remaining = [x for j, x in enumerate(arr) if j != i]
    
    max_possible_sum = 60 * 256
    s2c = [-1] * (max_possible_sum + 1)
    s2c[0] = 0
    max_sum = 0
    
    # Rebuild DP for remaining elements
    for x in remaining:
        new_entries = []
        valid = True
        for s in range(max_sum + 1):
            if s2c[s] >= 0:
                ns = s + x
                nk = s2c[s] + 1
                if s2c[ns] >= 0 and s2c[ns] != nk:
                    valid = False
                    break
                new_entries.append((ns, nk))
        if not valid:
            return False, None, None
        for ns, nk in new_entries:
            if s2c[ns] < 0 or s2c[ns] == nk:
                s2c[ns] = nk
        max_sum += x
    
    # Now try to add new_val
    success, new_s2c, new_max = _try_add(remaining, new_val, s2c, max_sum)
    if not success:
        return False, None, None
    
    # Reconstruct the set
    new_arr = remaining + [new_val]
    return True, new_s2c, new_max

def _swap_search(arr, deadline):
    """Local search using swaps to improve the set."""
    if len(arr) == 0:
        return arr
    
    max_possible_sum = 60 * 256
    sum_to_card = [-1] * (max_possible_sum + 1)
    sum_to_card[0] = 0
    current_max_sum = 0
    
    # Build DP for current set
    for x in arr:
        new_entries = []
        for s in range(current_max_sum + 1):
            if sum_to_card[s] >= 0:
                ns = s + x
                nk = sum_to_card[s] + 1
                sum_to_card[ns] = nk
        current_max_sum += x
    
    candidates = [x for x in range(1, 257) if x not in arr]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(arr)):
            if time.time() > deadline:
                break
            for new_val in candidates[:20]:  # Limit to first 20 candidates per position
                if new_val in arr:
                    continue
                success, new_s2c, new_max = _try_swap(arr, sum_to_card, current_max_sum, i, new_val)
                if success:
                    # Check if this swap is beneficial (for now, just accept any valid swap)
                    # We could add a heuristic here
                    arr[i] = new_val
                    sum_to_card = new_s2c
                    current_max_sum = new_max
                    improved = True
                    break
    
    return arr

def construct():
    deadline = time.time() + 80  # Safety margin
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    
    orderings = []
    orderings.append(list(range(1, 257)))
    orderings.append(list(range(256, 0, -1)))
    orderings.append([x for x in range(1, 257, 2)] + [x for x in range(2, 257, 2)])
    orderings.append([x for x in range(2, 257, 2)] + [x for x in range(1, 257, 2)])
    alt = []
    for i in range(128):
        alt.append(1 + i)
        alt.append(129 + i)
    orderings.append(alt)
    
    # Add more structured orderings
    for start in range(1, 257, 16):
        o = []
        for i in range(128):
            v1 = start + i
            v2 = start + 128 + i
            if v1 <= 256:
                o.append(v1)
            if v2 <= 256:
                o.append(v2)
        orderings.append(o)
    
    # Add random orderings
    num_random = 30
    for _ in range(num_random):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    
    # Greedy phase
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        current_set = _grow_greedy(ordering, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Local search phase with remaining time
    if best_size > 0 and time.time() < deadline - 5:
        improved_set = _swap_search(best_set[:], deadline - 2)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
