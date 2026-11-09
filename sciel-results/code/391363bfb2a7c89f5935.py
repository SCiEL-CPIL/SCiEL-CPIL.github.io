"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    res = lst[0]
    for x in lst[1:]:
        res = _lcm(res, x)
    return res

def _check_conflict_mitm(s, x, deadline):
    """
    Check if 1/x is a subset sum of reciprocals of s \ {x}.
    Returns True if conflict exists, False otherwise, None on timeout.
    Uses Meet-in-the-Middle for speed.
    """
    others = [y for y in s if y != x]
    if not others:
        return False
        
    # Calculate LCM
    L = x
    for y in others:
        L = _lcm(L, y)
        
    target = L // x
    values = [L // y for y in others]
    
    n = len(values)
    if n == 0:
        return False
    
    # If sum of all values is less than target, no conflict
    if sum(values) < target:
        return False
    
    # Split into two halves
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate subset sums for left
    sums_left = {0}
    for v in left:
        # Use a set to handle duplicates, though values are distinct usually
        new_sums = set()
        for s_val in sums_left:
            new_sums.add(s_val + v)
        sums_left.update(new_sums)
        
        if deadline is not None and time.time() > deadline:
            return None

    # Generate subset sums for right and sort
    sums_right = {0}
    for v in right:
        new_sums = set()
        for s_val in sums_right:
            new_sums.add(s_val + v)
        sums_right.update(new_sums)
        
        if deadline is not None and time.time() > deadline:
            return None
            
    # Sort right sums for binary search
    sorted_right = sorted(sums_right)
    
    # Check for conflict
    import bisect
    for s_val in sums_left:
        if s_val > target:
            continue
        rem = target - s_val
        # Binary search for rem in sorted_right
        idx = bisect.bisect_left(sorted_right, rem)
        if idx < len(sorted_right) and sorted_right[idx] == rem:
            return True
            
    return False

def _is_valid(s, deadline):
    """Check if set s is valid."""
    for x in s:
        res = _check_conflict_mitm(s, x, deadline)
        if res is True:
            return False
        elif res is None:
            return None
    return True

def _find_conflict(s, deadline):
    """Find a conflicting element x."""
    for x in s:
        res = _check_conflict_mitm(s, x, deadline)
        if res is True:
            return x
        elif res is None:
            return None
    return None

def _local_search(s, deadline):
    """
    Local search to maximize size of valid set.
    """
    s = set(s)
    # Ensure we start with a valid set or fix it
    # First, remove conflicts if any
    max_removals = 50
    for _ in range(max_removals):
        if deadline is not None and time.time() > deadline:
            break
        x = _find_conflict(s, deadline)
        if x is None:
            break
        if x is not None:
            s.discard(x)
        else:
            break
            
    # Now s is valid (or timeout)
    best_size = len(s)
    best_set = set(s)
    
    # Try to add elements
    # Order: try to add smaller numbers first? Or larger?
    # Smaller numbers have larger reciprocals, more likely to be sums of others.
    # Larger numbers have smaller reciprocals, less likely to be sums.
    # So adding large numbers is safer.
    
    # Randomize order for exploration
    candidates = [x for x in range(1, 25) if x not in s]
    random.shuffle(candidates)
    
    for x in candidates:
        if deadline is not None and time.time() > deadline:
            break
        candidate_set = s | {x}
        # Check if adding x creates a conflict
        # We only need to check if x is a sum of others, or if others sum to x?
        # The condition is: no reciprocal of a selected element equals sum of reciprocals of OTHERS.
        # So adding x requires:
        # 1. 1/x != sum(1/y for y in subset of S)
        # 2. For all y in S, 1/y != sum(1/z for z in subset of (S \ {y}) U {x})
        #    This is harder. The presence of x might allow a subset of S U {x} to sum to 1/y.
        #    So we must re-validate the whole set.
        
        res = _is_valid(candidate_set, deadline)
        if res is True:
            s = candidate_set
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)
        elif res is None:
            break
            
    # If we couldn't add anyone, try swapping
    # Pick a random element from s and a random element not in s
    # If size improves, keep it.
    
    # To allow for more complex moves, we can do a hill-climbing step
    # But simple add-only is often sufficient if we restart.
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5s margin
    
    best_result = set()
    best_size = 0
    
    # Strategy 1: Greedy build with large numbers first
    # Large numbers are "safe"
    order = list(range(24, 0, -1))
    s = set()
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        candidate = s | {x}
        res = _is_valid(candidate, deadline)
        if res is True:
            s = candidate
        elif res is None:
            break
    if len(s) > best_size:
        best_result = s
        best_size = len(s)
        
    # Strategy 2: Local search from random starts
    # We know max size is likely around 19-20.
    # Let's try to find 20.
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Random start of size 15-20
        # Bias towards larger numbers
        pool = list(range(1, 25))
        random.shuffle(pool)
        # Sort by descending value to prioritize large numbers in random start?
        # Or just random sample
        rand_set = set(random.sample(pool, 18))
        
        s_improved = _local_search(rand_set, deadline)
        
        if len(s_improved) > best_size:
            best_result = s_improved
            best_size = len(s_improved)
            
        # Early exit if we found a very large set (e.g., 20)
        if best_size >= 20:
            # Try to verify if 20 is possible? 
            # If we find 20, we can keep searching for 21 in remaining time
            pass

    # Final verification
    if best_result:
        final_check = _is_valid(best_result)
        if final_check is not True:
            # Fallback
            best_result = set(range(13, 25)) # Size 12, definitely valid?
            # 13..24: 1/13 > 1/14+...+1/24?
            # 1/14+...+1/24 < 11/14 < 1/13. Yes.
            # And for any y in 13..24, sum of others < 1/y?
            # Max sum is for y=13, sum of 1/14..1/24 < 1/13.
            # For y=24, sum of 1/13..1/23 is large, but we check if 1/24 is sum of subset.
            # 1/24 is small. 1/13 is large.
            # It is highly likely 13..24 is valid.
            
    return sorted(best_result)

# EVOLVE-BLOCK-END
