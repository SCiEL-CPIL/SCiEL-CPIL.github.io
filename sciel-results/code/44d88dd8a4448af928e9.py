"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def _check_divisibility_conflict(new_sums, existing_sums):
    """
    Check if any new sum conflicts with existing sums or other new sums via divisibility.
    Returns True if conflict exists.
    """
    # Check new sums against each other for divisibility
    new_list = sorted(new_sums)
    for i in range(len(new_list)):
        a = new_list[i]
        if a <= 0:
            continue
        for j in range(i+1, len(new_list)):
            b = new_list[j]
            if b % a == 0:
                return True
        # Check new sum against existing sums
        for e in existing_sums:
            # Check if new sum divides existing or vice versa
            if e > 0:
                if e % a == 0 or (a > 0 and a % e == 0):
                    return True
    return False

def _try_add_incremental(arr, sums_set, cand):
    """
    Try to add cand to arr. Returns (success, new_sums_set) if successful.
    Uses incremental computation: new sums = old sums ∪ {cand} ∪ {s+cand for s in old sums}
    """
    if cand in arr:
        return False, None
    
    # Compute new sums
    old_sums = list(sums_set)
    new_sums = {cand}
    for s in old_sums:
        ns = s + cand
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
    
    # Check for duplicates with existing sums
    for ns in new_sums:
        if ns in sums_set:
            return False, None
    
    # Check divisibility: new sums vs existing sums
    for ns in new_sums:
        for es in sums_set:
            if es > 0 and ns > 0:
                if ns % es == 0 or es % ns == 0:
                    return False, None
    
    # Check divisibility among new sums
    new_list = sorted(new_sums)
    for i in range(len(new_list)):
        a = new_list[i]
        for j in range(i+1, len(new_list)):
            b = new_list[j]
            if b % a == 0:
                return False, None
    
    # Success: merge
    new_sums_set = sums_set | new_sums
    return True, new_sums_set

def _compute_all_sums(arr):
    """Compute all non-empty subset sums of arr."""
    n = len(arr)
    sums_set = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums_set:
            return None  # duplicate sum
        sums_set.add(s)
    return sums_set

def _check_divisibility(sums_set):
    """Check if any two distinct values in sums_set divide one another."""
    sorted_sums = sorted(sums_set)
    n = len(sorted_sums)
    if n == 0:
        return False
    max_sum = sorted_sums[-1]
    
    for i in range(n):
        si = sorted_sums[i]
        if si <= 0:
            continue
        multiple = 2 * si
        while multiple <= max_sum:
            if multiple in sums_set:
                return True
            multiple += si
    return False

def _is_valid(arr):
    """Check if arr is a valid set."""
    if not arr:
        return False
    if len(set(arr)) != len(arr):
        return False
    for x in arr:
        if not (isinstance(x, int) and 1 <= x <= 256):
            return False
    
    sums_set = _compute_all_sums(arr)
    if sums_set is None:
        return False
    if _check_divisibility(sums_set):
        return False
    return True

def _greedy_build(candidates, deadline):
    """
    Build a valid set greedily from candidates using incremental sum tracking.
    Returns the constructed set.
    """
    current = []
    sums_set = set()
    used = set()
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in used:
            continue
        
        success, new_sums = _try_add_incremental(current, sums_set, cand)
        if success:
            current.append(cand)
            sums_set = new_sums
            used.add(cand)
    
    return current

def _greedy_extend(base_arr, base_sums, candidates, deadline):
    """
    Extend a base set with candidates using incremental sum tracking.
    Returns (extended_arr, extended_sums).
    """
    current = list(base_arr)
    sums_set = set(base_sums)
    used = set(current)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in used:
            continue
        
        success, new_sums = _try_add_incremental(current, sums_set, cand)
        if success:
            current.append(cand)
            sums_set = new_sums
            used.add(cand)
    
    return current, sums_set

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Phase 1: Deterministic Greedy with different orderings
    orderings = [
        list(range(256, 0, -1)),
        list(range(1, 257)),
        list(range(128, 257)) + list(range(1, 129)),
        list(range(1, 129)) + list(range(128, 257)),
        list(range(192, 257)) + list(range(128, 193)) + list(range(64, 129)) + list(range(1, 65)),
        list(range(1, 65)) + list(range(64, 129)) + list(range(128, 193)) + list(range(192, 257)),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = _greedy_build(order, deadline)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Phase 2: Randomized Greedy Restarts with incremental tracking
    all_nums = list(range(1, 257))
    large_nums = list(range(129, 257))
    small_nums = list(range(1, 129))
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        r = rng.random()
        if r < 0.25:
            candidates = large_nums + small_nums
        elif r < 0.5:
            candidates = small_nums + large_nums
        elif r < 0.75:
            candidates = list(range(1, 257))
        else:
            # Weighted toward larger numbers
            candidates = [x for x in range(1, 257) if rng.random() < 0.3 or x > 100]
        
        rng.shuffle(candidates)
        
        current = _greedy_build(candidates, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Phase 3: Local Search - Try to improve best_set
    if best_size < 30 and time.time() < deadline:
        # Compute sums for best_set
        best_sums = _compute_all_sums(best_set)
        if best_sums is None:
            best_sums = set()
        
        used_in_best = set(best_set)
        remaining_candidates = [x for x in range(1, 257) if x not in used_in_best]
        rng.shuffle(remaining_candidates)
        
        # Try to add any remaining candidates
        improved, improved_sums = _greedy_extend(best_set, best_sums, remaining_candidates, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
        
        # Try targeted swaps: replace one element with another
        for _ in range(2000):
            if time.time() > deadline:
                break
            if not best_set:
                break
            
            idx = rng.randint(0, len(best_set) - 1)
            old_val = best_set[idx]
            
            # Try replacing with a value from remaining or slightly perturbed
            if rng.random() < 0.5 and remaining_candidates:
                new_val = rng.choice(remaining_candidates)
            else:
                delta = rng.randint(-20, 20)
                new_val = old_val + delta
                if new_val < 1:
                    new_val = 1
                if new_val > 256:
                    new_val = 256
            
            if new_val in used_in_best:
                continue
            
            # Build trial set
            trial = best_set[:]
            trial[idx] = new_val
            trial.sort()
            
            # Check validity
            trial_sums = _compute_all_sums(trial)
            if trial_sums is None:
                continue
            if _check_divisibility(trial_sums):
                continue
            
            # Try to extend
            used_in_trial = set(trial)
            rem = [x for x in range(1, 257) if x not in used_in_trial]
            rng.shuffle(rem)
            
            extended, _ = _greedy_extend(trial, trial_sums, rem, deadline)
            if len(extended) > best_size:
                best_set = extended
                best_size = len(best_set)
                used_in_best = set(best_set)
                remaining_candidates = [x for x in range(1, 257) if x not in used_in_best]
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
