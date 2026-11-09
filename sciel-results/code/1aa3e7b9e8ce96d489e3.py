"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums_with_counts(elements):
    """Return dict of sum -> count using incremental DP."""
    n = len(elements)
    sums = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        sums[mask] = sums[mask ^ lsb] + elements[i]
    sum_count = {}
    for mask in range(1, 1 << n):
        s = sums[mask]
        sum_count[s] = sum_count.get(s, 0) + 1
    return sum_count

def _check_divisibility_fast(sums_list):
    """Check no two distinct sums divide each other."""
    s = sorted(sums_list)
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            if s[j] % si == 0:
                return False
    return True

def _is_valid_fast(elements):
    """Fast validity check with incremental sums."""
    n = len(elements)
    if n == 0:
        return True
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    sum_count = _subset_sums_with_counts(elements)
    for s, c in sum_count.items():
        if c > 1:
            return False
    sums_list = list(sum_count.keys())
    if not _check_divisibility_fast(sums_list):
        return False
    return True

def _try_add_incremental(current, sums_dict, candidate, deadline):
    """Try adding candidate using precomputed sums. Returns (new_sums_dict, True) or (None, False)."""
    if time.time() > deadline:
        return None, False
    new_sums_dict = {}
    for s, c in sums_dict.items():
        new_sums_dict[s] = new_sums_dict.get(s, 0) + c
    new_sums_dict[candidate] = new_sums_dict.get(candidate, 0) + 1
    for s, c in sums_dict.items():
        ns = s + candidate
        new_sums_dict[ns] = new_sums_dict.get(ns, 0) + c
    for s, c in new_sums_dict.items():
        if c > 1:
            return None, False
    sums_list = list(new_sums_dict.keys())
    if not _check_divisibility_fast(sums_list):
        return None, False
    return new_sums_dict, True

def _greedy_build(candidates, deadline, max_size=30):
    """Greedy: add candidates in given order if valid, using incremental sums."""
    current = []
    sums_dict = {}
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        new_sums, ok = _try_add_incremental(current, sums_dict, c, deadline)
        if ok:
            current.append(c)
            sums_dict = new_sums
    return current

def _local_search_improve(best_set, deadline):
    """Try to improve by adding elements using incremental sums."""
    current = list(best_set)
    sums_dict = _subset_sums_with_counts(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current:
                continue
            new_sums, ok = _try_add_incremental(current, sums_dict, c, deadline)
            if ok:
                current.append(c)
                sums_dict = new_sums
                improved = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 75
    best_set = [31, 32, 34]
    best_len = len(best_set)
    
    # Strategy 1: Greedy with large numbers first (descending)
    cands_desc = list(range(256, 0, -1))
    result = _greedy_build(cands_desc, deadline)
    if len(result) > best_len:
        best_set = result
        best_len = len(best_set)
    
    # Strategy 2: Multiple randomized greedy attempts
    for attempt in range(200):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        result = _greedy_build(cands, deadline)
        if len(result) > best_len:
            best_set = result
            best_len = len(best_set)
            if best_len >= 30:
                break
    
    # Strategy 3: Local search improvement
    if time.time() < deadline:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(best_set)
    
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
            if not _is_valid_fast(ss):
                continue
            cands = [c for c in range(1, 257) if c not in ss]
            random.shuffle(cands)
            current = list(ss)
            sums_dict = _subset_sums_with_counts(current)
            for c in cands:
                if time.time() > deadline:
                    break
                if len(current) >= 30:
                    break
                new_sums, ok = _try_add_incremental(current, sums_dict, c, deadline)
                if ok:
                    current.append(c)
                    sums_dict = new_sums
            if len(current) > best_len:
                best_set = current
                best_len = len(best_set)
                if best_len >= 30:
                    break
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(best_set)
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
