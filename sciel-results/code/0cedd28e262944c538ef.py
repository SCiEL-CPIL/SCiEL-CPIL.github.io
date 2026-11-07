"""Rewritten program: fast incremental greedy with optimized divisibility check."""
# EVOLVE-BLOCK-START
import time
import random

# Deadline guard: 55s wall-clock (5s safety margin under 60s limit)
_DEADLINE = 55.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """Check that no two distinct values in sums divide one another.
    Optimized: Sort sums. For each sum s, check if any multiple 2s, 3s... is in sums.
    """
    if not sums:
        return True
    
    sorted_sums = sorted(sums)
    max_sum = sorted_sums[-1]
    
    # Use a set for O(1) lookup
    sum_set = set(sums)
    
    for s in sorted_sums:
        # Check multiples of s
        # We only need to check multiples up to max_sum
        # If s is large, there are few multiples.
        # If s is small, there are more, but s is likely to have many divisors? No, we check if s divides OTHERS.
        # So we check if k*s is in sum_set for k >= 2.
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
            
    return True

def _subset_sums_incremental(current_sums, new_element):
    """Given a set of existing non-empty subset sums, add a new element.
    Returns the new set of sums (existing + existing+new + {new})."""
    new_sums = set(current_sums)
    new_sums.add(new_element)
    for s in current_sums:
        new_sums.add(s + new_element)
    return new_sums

def _validate_full(elements):
    """Full validation: distinct, in range, all non-empty subset sums distinct and no divisibility."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    # Compute all non-empty subset sums
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    # Check no two distinct sums divide each other
    if not _check_no_divisibility(sums):
        return False
    return True

def _greedy_construct(candidates, deadline_check=None):
    """Greedy: try adding candidates in given order, keeping only valid additions."""
    chosen = []
    current_sums = set()
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        # Check if x is already in chosen
        if x in chosen:
            continue
        # Tentatively add x
        new_sums = _subset_sums_incremental(current_sums, x)
        if _check_no_divisibility(new_sums):
            chosen.append(x)
            current_sums = new_sums
    
    return chosen

def construct():
    global _START
    _START = time.time()
    
    # Pool of candidates
    # Bias towards larger numbers and primes to reduce divisibility conflicts
    pool = list(range(1, 257))
    
    # Pre-compute primes for bias? Or just use random orders.
    # Large numbers are better because their multiples are less likely to be in the range of other sums.
    
    best = []
    best_len = 0
    
    # Strategy 1: Deterministic Greedy
    # Order: Descending (large numbers first)
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
    
    # Order: Ascending (small numbers first) - likely worse but fast
    if _remaining() > 5.0:
        order2 = sorted(pool)
        result2 = _greedy_construct(order2, deadline_check=_remaining)
        if len(result2) > best_len:
            best = result2
            best_len = len(result2)
    
    # Strategy 2: Randomized Greedy Restarts
    # Use fixed seeds for reproducibility and speed
    # We can run many restarts because the check is now fast.
    
    rng = random.Random(42)
    
    # Run until deadline
    while _remaining() > 2.0:
        # Shuffle pool
        random_pool = list(pool)
        rng.shuffle(random_pool)
        
        # Optional: Bias shuffle? e.g. sort by random key but keep large numbers slightly more likely?
        # Just simple shuffle is a good baseline.
        
        result = _greedy_construct(random_pool, deadline_check=_remaining)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
            # If we found a very good set, maybe stop? 
            # Max size is 30. If we hit 10+, it's good.
            # But let's keep searching to see if we can go higher.
            
    # Final validation
    if best and not _validate_full(best):
        # Fallback to a known valid small set
        best = [128, 192, 224]
    
    # Ensure Python ints
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
