import time

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    max_sum = sum(arr)
    if max_sum > 256 * n:
        # Still need to check, but sums are spread out
        pass
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    # Check pairwise disjointness with early exit
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def _find_best_ap(deadline):
    """Find the largest valid AP, trying multiple starting positions."""
    best = [1]
    for k in range(60, 1, -1):
        if time.time() > deadline - 30:
            break
        # For AP {x, x+1, ..., x+k-1}, the constraint is that
        # sum of j smallest < sum of (j+1) largest for all j
        # Actually we need: for any two subsets of different sizes, sums differ
        # The critical case is: sum of j largest from size-j subset vs sum of (j+1) smallest from size-(j+1) subset
        # For consecutive integers, subset of size j has max sum = x+(x+1)+...+(x+k-j) = j*x + j*(k-j)*... 
        # Let's just try several starting values
        found = False
        # The AP must fit in 1..256: x >= 1, x+k-1 <= 256 => x <= 256-k+1
        max_x = 256 - k + 1
        # Try starting values from high to low (larger values spread sums more)
        for x in range(max_x, 0, -1):
            if time.time() > deadline - 25:
                break
            vals = list(range(x, x + k))
            if _is_valid_bitset(vals):
                best = vals
                found = True
                break
        if found:
            break
    return best

def _greedy_extend(base_arr, deadline):
    """Try to extend base_arr by adding elements from 1..256 not already in it."""
    current = list(base_arr)
    current_set = set(current)
    # Sort candidates: try smaller values first as they're less likely to cause conflicts
    # Actually, let's try both orders and pick the better
    candidates = [v for v in range(1, 257) if v not in current_set]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for v in candidates:
            if time.time() > deadline - 5:
                break
            trial = current + [v]
            if _is_valid_bitset(trial):
                current = trial
                current_set.add(v)
                candidates.remove(v)
                improved = True
                break
    return current

def _greedy_extend_desc(base_arr, deadline):
    """Try to extend base_arr by adding elements from 256 down to 1 not already in it."""
    current = list(base_arr)
    current_set = set(current)
    candidates = [v for v in range(256, 0, -1) if v not in current_set]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for v in candidates:
            if time.time() > deadline - 5:
                break
            trial = current + [v]
            if _is_valid_bitset(trial):
                current = trial
                current_set.add(v)
                candidates.remove(v)
                improved = True
                break
    return current

def _local_search(base_arr, deadline):
    """Try to improve the set by swapping elements to allow more additions."""
    current = list(base_arr)
    current_set = set(current)
    best = current
    
    while time.time() < deadline - 5:
        found_swap = False
        for i, elem in enumerate(current):
            if time.time() > deadline - 5:
                break
            candidates = [v for v in range(1, 257) if v not in current_set]
            for v in candidates:
                if time.time() > deadline - 5:
                    break
                trial = current[:i] + [v] + current[i+1:]
                if _is_valid_bitset(trial):
                    trial_set = set(trial)
                    trial_candidates = [w for w in range(1, 257) if w not in trial_set]
                    extended = list(trial)
                    for w in trial_candidates:
                        if time.time() > deadline - 5:
                            break
                        ext_trial = extended + [w]
                        if _is_valid_bitset(ext_trial):
                            extended = ext_trial
                            trial_candidates.remove(w)
                    if len(extended) > len(best):
                        best = extended
                        current = extended
                        current_set = set(current)
                        found_swap = True
                        break
            if found_swap:
                break
        if not found_swap:
            break
    
    return best

def construct():
    deadline = time.time() + 85
    
    # Find the largest valid AP
    best = _find_best_ap(deadline)
    
    # Try to extend with greedy (ascending order)
    if len(best) < 60 and time.time() < deadline - 10:
        ext1 = _greedy_extend(best, deadline)
        if len(ext1) > len(best):
            best = ext1
    
    # Try to extend with greedy (descending order) from original AP
    if len(best) < 60 and time.time() < deadline - 10:
        ext2 = _greedy_extend_desc(_find_best_ap(deadline), deadline)
        if len(ext2) > len(best):
            best = ext2
    
    # Try local search to improve further
    if len(best) < 60 and time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    return list(best)
