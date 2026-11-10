"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations
import random

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _get_sums(S):
    """Compute all nonempty subset sums of S. Returns set or None if collision."""
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
    if not sums_set:
        return True
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _is_valid_set(S):
    """
    Check if S (list of distinct ints) satisfies:
    1) All nonempty subset sums are distinct (no two different subsets have same sum).
    2) No two distinct nonempty subset sums divide one another.
    """
    if len(set(S)) != len(S):
        return False
    if any(not isinstance(x, int) or isinstance(x, bool) for x in S):
        return False
    sums = _get_sums(S)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _extend_greedy(current_set, max_val=256):
    """
    Try to find a single integer x not in current_set, 1 <= x <= max_val,
    such that current_set + [x] is valid. Returns x or None.
    """
    base_sums = _get_sums(current_set)
    if base_sums is None:
        return None
    
    # Precompute existing elements for quick check
    existing = set(current_set)
    
    # We need to check x from 1 to max_val
    # Optimization: If we have a set of sums, adding x creates new sums {x, x+s for s in base_sums}.
    # These must not collide with each other or base_sums.
    # And the combined set must be divisibility-free.
    
    # To speed up, we can iterate x in a smart order?
    # Usually, smaller x are more likely to cause collisions or divisibility.
    # But larger x might be "safer" for divisibility if they are large primes?
    # However, the range is only 1..256.
    
    for x in range(1, max_val + 1):
        if x in existing:
            continue
        
        # Check for sum collisions
        # New sums are {x} U {x + s | s in base_sums}
        # Collision if:
        # 1. x in base_sums
        # 2. x + s1 == x + s2 for s1 != s2 (impossible since s1 != s2)
        # 3. x + s1 == s2 for some s1, s2 in base_sums
        # 4. x + s1 == x (impossible since s1 > 0)
        
        if x in base_sums:
            continue
            
        # Check 3: x + s1 == s2 => s2 - s1 == x
        # This is equivalent to checking if x is the difference of two subset sums.
        # Since base_sums can be large, let's just compute the new sums set and check for duplicates.
        # But computing the full set for every x is O(2^n).
        # Let's just do it. n is small (<= 10 or so).
        
        new_sums = set(base_sums)
        valid_distinct = True
        for s in base_sums:
            ns = s + x
            if ns in new_sums:
                valid_distinct = False
                break
            new_sums.add(ns)
        
        if not valid_distinct:
            continue
        
        # Check divisibility
        if _check_divisibility(new_sums):
            return x
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Known valid set from seed
    best_set = [31, 32, 34]
    
    # Strategy 1: Greedy extension from the known set
    current = list(best_set)
    while time.time() < deadline - 15:
        x = _extend_greedy(current)
        if x is None:
            break
        current.append(x)
        if len(current) >= 30:
            break
    
    if len(current) > len(best_set) and _is_valid_set(current):
        best_set = current

    # Strategy 2: Random search for larger sets
    # We can also try to find sets of specific sizes k, starting from len(best_set)+1
    # To make random search more effective, we can bias towards "large" numbers or "sparse" numbers.
    # But simple random sampling is a good baseline.
    
    def random_search(k, max_val=256, iterations=1000):
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            # Sample k distinct integers from 1..max_val
            S = sorted(random.sample(range(1, max_val + 1), k))
            if _is_valid_set(S):
                return S
        return None

    # Try to find sets larger than best_set
    target_k = len(best_set) + 1
    while target_k <= 30 and time.time() < deadline - 5:
        # Adjust iterations based on remaining time
        remaining = deadline - 5 - time.time()
        # Assume ~1ms per check (optimistic). If it's slower, we do fewer.
        # Let's just do a fixed number of iterations per k, but break early if time runs out.
        S = random_search(target_k, iterations=500)
        if S and len(S) > len(best_set):
            best_set = S
            target_k += 1
        else:
            # If we failed to find a set of size k, it's likely k is too hard or we just got unlucky.
            # Try a few more times with different iterations?
            # Or move on to k+1? No, if we can't find k, finding k+1 is harder.
            # But maybe we just need more luck.
            # Let's try a few more random restarts for the same k.
            S = random_search(target_k, iterations=500)
            if S and len(S) > len(best_set):
                best_set = S
                target_k += 1
            else:
                # Give up on this k and try a slightly smaller one? No, we want max size.
                # Let's just stop random search if we can't grow.
                break

    # Final validation
    if _is_valid_set(best_set):
        # Ensure types are correct
        return [int(x) for x in best_set]
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
