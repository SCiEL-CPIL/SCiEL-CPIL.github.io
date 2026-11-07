import time
import random
from itertools import combinations

def _check_divisibility_sorted(sorted_sums):
    """Check if any two values in sorted_sums have divisibility relation."""
    n = len(sorted_sums)
    for i in range(n):
        si = sorted_sums[i]
        if si == 0:
            continue
        # For each j > i, check if sorted_sums[j] % si == 0
        # Optimization: if sorted_sums[j] >= 2*si, it could be a multiple
        for j in range(i + 1, n):
            sj = sorted_sums[j]
            if sj >= 2 * si:
                if sj % si == 0:
                    return False
            else:
                break  # sj < 2*si, and since sorted, all further sj will also be < 2*si or larger but we need to check
    return True

def _valid_set(nums):
    """Full validation of a set of numbers."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        # Too many subset sums to enumerate efficiently
        # Use heuristic: check if all numbers are large and "independent"
        # For now, return False to avoid timeout
        return False
    
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return False
        sums.add(s)
    
    sorted_sums = sorted(sums)
    return _check_divisibility_sorted(sorted_sums)

def _try_greedy_extend(current_set, current_sums, candidates, deadline, max_size=30):
    """Greedy extension: try to add candidates in order, keeping valid ones."""
    current_set = list(current_set)
    current_sums = set(current_sums)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(current_set) >= max_size:
            break
        
        # Check if cand can be added
        # 1. Check for duplicate sums
        if cand in current_sums:
            continue
        
        new_sums = set(current_sums)
        valid = True
        for s in current_sums:
            ns = s + cand
            if ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
        if not valid:
            continue
        new_sums.add(cand)
        
        # 2. Check divisibility - only check new sums against all sums
        # New sums: {cand} and {s+cand for s in current_sums}
        # Old sums: current_sums
        
        # Check cand against current_sums
        for s in current_sums:
            if cand > s and cand % s == 0:
                valid = False
                break
            if s > cand and s % cand == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Check (s+cand) against current_sums (t)
        # (s+cand) % t == 0 or t % (s+cand) == 0
        for s in current_sums:
            sc = s + cand
            for t in current_sums:
                if sc > t and sc % t == 0:
                    valid = False
                    break
                if t > sc and t % sc == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check (s+cand) against cand: (s+cand) % cand == 0 => s % cand == 0
        for s in current_sums:
            if s % cand == 0:
                valid = False
                break
        if not valid:
            continue
        
        current_set.append(cand)
        current_sums = new_sums
    
    return current_set, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Strategy: Use backtracking with forward-checking
    # Start with a good initial set, then try to extend
    
    # Initial candidates: primes in [129, 256] are good starting points
    # But we should also try different strategies
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy from large primes
    primes_large = [p for p in range(129, 257) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    primes_large.sort(reverse=True)
    
    set1, sums1 = _try_greedy_extend([], set(), primes_large, deadline, max_size=30)
    if len(set1) > best_size and _valid_set(set1):
        best_set = set1
        best_size = len(set1)
    
    if time.time() > deadline:
        return best_set if best_set else [31, 32, 34]
    
    # Strategy 2: Try different starting points
    # Use numbers that are "independent" - e.g., powers of 2 times large primes
    # Or use a randomized approach
    
    random.seed(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Randomly select a subset of candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Sort by some heuristic: prefer numbers with fewer divisors
        def divisor_count(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count
        
        candidates.sort(key=lambda x: (divisor_count(x), -x))
        
        set_a, sums_a = _try_greedy_extend([], set(), candidates, deadline, max_size=30)
        if len(set_a) > best_size and _valid_set(set_a):
            best_set = set_a
            best_size = len(set_a)
    
    # Strategy 3: Try to improve best_set by swapping elements
    if best_set and best_size < 30:
        if time.time() < deadline - 10:
            best_set, best_sums = _try_greedy_extend(best_set, set(), 
                                                     [i for i in range(1, 257) if i not in best_set],
                                                     deadline, max_size=30)
            if _valid_set(best_set):
                best_size = len(best_set)
    
    # Final validation
    if best_set and _valid_set(best_set):
        return best_set
    elif best_set:
        # Try to trim to a valid subset
        for size in range(len(best_set), 0, -1):
            subset = best_set[:size]
            if _valid_set(subset):
                return subset
        return [31, 32, 34]
    else:
        return [31, 32, 34]
