"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_subset_sums_with_counts(elements):
    """Return dict: sum -> count of subsets producing that sum."""
    n = len(elements)
    sum_count = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sum_count[s] = sum_count.get(s, 0) + 1
    return sum_count

def _check_divisibility(sums):
    """Check no two distinct sums divide each other. Returns True if valid."""
    s = sorted(sums)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[j] % s[i] == 0:
                return False
    return True

def _is_valid(elements):
    """Full validity check."""
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    sum_count = _get_subset_sums_with_counts(elements)
    if any(v > 1 for v in sum_count.values()):
        return False
    return _check_divisibility(list(sum_count.keys()))

def _try_extend(current, candidate, deadline):
    """Try adding candidate to current list, return new list or None."""
    if time.time() > deadline:
        return None
    new_list = current + [candidate]
    if _is_valid(new_list):
        return new_list
    return None

def _greedy_build(candidates, deadline, max_size=30):
    """Greedy: add candidates in given order if valid."""
    current = []
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        new_list = _try_extend(current, c, deadline)
        if new_list is not None:
            current = new_list
    return current

def _local_search_improve(best_set, deadline):
    """Try to improve by adding or swapping elements."""
    current = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a new element
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current:
                continue
            new_list = current + [c]
            if _is_valid(new_list):
                current = new_list
                improved = True
                break
        # Try single swap: remove one element, add another
        if not improved:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for c in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c in current:
                        continue
                    new_list = current[:i] + [c] + current[i+1:]
                    if _is_valid(new_list):
                        # After swap, try to add one more
                        for c2 in range(1, 257):
                            if time.time() > deadline:
                                break
                            if c2 in new_list:
                                continue
                            if _is_valid(new_list + [c2]):
                                current = new_list + [c2]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    
    # Strategy 1: Greedy with large numbers first (descending)
    cands_desc = list(range(256, 0, -1))
    result = _greedy_build(cands_desc, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # Strategy 2: Multiple randomized greedy attempts
    for attempt in range(200):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        result = _greedy_build(cands, deadline)
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= 30:
                break
    
    # Strategy 3: Local search improvement
    if time.time() < deadline:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Strategy 4: Try specific known-good starting points and extend
    if time.time() < deadline:
        starting_sets = [
            [128, 192, 224],
            [100, 150, 200],
            [64, 96, 128, 160],
            [80, 120, 160, 200, 240],
            [50, 80, 110, 140, 170, 200],
        ]
        for ss in starting_sets:
            if time.time() > deadline:
                break
            if not _is_valid(ss):
                continue
            cands = [c for c in range(1, 257) if c not in ss]
            random.shuffle(cands)
            current = list(ss)
            for c in cands:
                if time.time() > deadline:
                    break
                if len(current) >= 30:
                    break
                new_list = _try_extend(current, c, deadline)
                if new_list is not None:
                    current = new_list
            if len(current) > len(best_set):
                best_set = current
                if len(best_set) >= 30:
                    break
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
