"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _get_sums_safe(S):
    """Compute all nonempty subset sums. Returns set or None if any collision."""
    sums = set()
    n = len(S)
    for r in range(1, n + 1):
        for comb in combinations(S, r):
            s = sum(comb)
            if s in sums:
                return None
            sums.add(s)
    return sums

def _check_divisibility(sums_set):
    """Check if no two distinct values in sums_set divide one another."""
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1] if sums_list else 0
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _is_valid_set(S):
    """Full validation of set S."""
    if len(set(S)) != len(S):
        return False
    sums = _get_sums_safe(S)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_add(S, x, base_sums):
    """Try adding x to S given base_sums. Returns new_sums set or None if invalid."""
    if x in S:
        return None
    new_sums = set(base_sums)
    for s in base_sums:
        ns = s + x
        if ns in new_sums:
            return None
        new_sums.add(ns)
    if not _check_divisibility(new_sums):
        return None
    return new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Start with a known valid small set
    best_set = [31, 32, 34]
    
    # Backtracking search to extend the set
    def backtrack(current_set, base_sums, candidates, depth, max_depth):
        """Try to extend current_set by adding from candidates."""
        nonlocal best_set
        if time.time() > deadline - 5:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        
        if depth >= max_depth or time.time() > deadline - 5:
            return
        
        # Try each candidate in order (larger first to reduce divisibility conflicts)
        for i in range(len(candidates)):
            if time.time() > deadline - 5:
                return
            x = candidates[i]
            new_sums = _try_add(current_set, x, base_sums)
            if new_sums is not None:
                backtrack(current_set + [x], new_sums, candidates[i+1:], depth + 1, max_depth)
    
    # Generate candidate pool: focus on larger numbers
    # Numbers > 128 are less likely to divide other sums
    candidates = list(range(256, 1, -1))  # Try larger numbers first
    
    # Phase 1: Greedy extension from known set
    current = list(best_set)
    base_sums = _get_sums_safe(current)
    if base_sums is not None:
        # Try to extend greedily, preferring larger numbers
        remaining = [x for x in candidates if x not in current]
        extended = list(current)
        ext_sums = base_sums
        for x in remaining:
            if time.time() > deadline - 20:
                break
            new_sums = _try_add(extended, x, ext_sums)
            if new_sums is not None:
                extended.append(x)
                ext_sums = new_sums
                if len(extended) >= 30:
                    break
        if len(extended) > len(best_set) and _is_valid_set(extended):
            best_set = extended
    
    # Phase 2: Backtracking search for larger sets
    # Try different starting points
    start_options = [
        [31, 32, 34],
        [50, 51, 53],
        [100, 101, 103],
        [128, 129, 131],
        [200, 201, 203],
    ]
    
    for start in start_options:
        if time.time() > deadline - 10:
            break
        if not _is_valid_set(start):
            continue
        start_sums = _get_sums_safe(start)
        if start_sums is None:
            continue
        remaining = [x for x in candidates if x not in start]
        backtrack(list(start), start_sums, remaining, 0, 25)
    
    # Phase 3: Random search with restarts if time remains
    import random
    if time.time() < deadline - 15:
        for k in range(len(best_set) + 1, min(len(best_set) + 5, 20)):
            if time.time() > deadline - 10:
                break
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Bias toward larger numbers
                pool = list(range(64, 257))
                S = sorted(random.sample(pool, k))
                if _is_valid_set(S):
                    if len(S) > len(best_set):
                        best_set = S
                    break
    
    # Final validation
    if _is_valid_set(best_set):
        return list(best_set)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
