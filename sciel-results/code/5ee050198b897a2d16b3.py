import time

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
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

def _ap_solution(k):
    """Return AP of size k if it fits in 1..256 and is valid, else None."""
    if k == 1:
        return [1]
    max_j_val = max(j * (k - 1 - j) for j in range(1, k))
    required_x_1 = max_j_val + 1
    if required_x_1 <= 256 - k + 1:
        x_1 = required_x_1
        set_vals = list(range(x_1, x_1 + k))
        if len(set(set_vals)) == k and all(1 <= v <= 256 for v in set_vals):
            if _is_valid_bitset(set_vals):
                return set_vals
    return None

def _greedy_extend(base_arr, deadline):
    """Try to extend base_arr by adding elements from 1..256 not already in it."""
    current = list(base_arr)
    current_set = set(current)
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

def _local_search(base_arr, deadline):
    """Try to improve the set by swapping elements to allow more additions."""
    current = list(base_arr)
    current_set = set(current)
    best = current
    
    while time.time() < deadline - 5:
        # Try swapping each element with a candidate
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
                    # Check if this swap allows us to add more elements
                    trial_set = set(trial)
                    trial_candidates = [w for w in range(1, 257) if w not in trial_set]
                    # Try to extend from this swapped state
                    extended = list(trial)
                    extended_set = set(extended)
                    for w in trial_candidates:
                        if time.time() > deadline - 5:
                            break
                        ext_trial = extended + [w]
                        if _is_valid_bitset(ext_trial):
                            extended = ext_trial
                            extended_set.add(w)
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
    deadline = time.time() + 80
    
    # Find the largest k for which an AP solution exists and is valid
    best = [1]
    for k in range(60, 0, -1):
        sol = _ap_solution(k)
        if sol is not None:
            best = sol
            break
    
    # Try to extend the AP solution with additional elements
    if len(best) < 60 and time.time() < deadline - 10:
        extended = _greedy_extend(best, deadline)
        if len(extended) > len(best):
            best = extended
    
    # Try local search to improve further
    if len(best) < 60 and time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    return list(best)
