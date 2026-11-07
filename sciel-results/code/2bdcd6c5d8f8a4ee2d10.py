import time
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin

def _is_valid_set(nums):
    """Full validity check for a list of numbers."""
    if not nums:
        return True
    
    n = len(nums)
    sums = set()
    # Generate all nonempty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return False # Duplicate sums are invalid
        sums.add(s)
        
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % si == 0:
                return False
    return True

def _can_add_fast(nums, new_elem, existing_sums):
    """
    Check if adding new_elem to nums (with known existing_sums) keeps the set valid.
    Returns (valid, new_sums) where new_sums is the updated sum set if valid.
    """
    if not nums:
        return True, {new_elem}
    
    # New sums = existing_sums U {new_elem} U {s + new_elem for s in existing_sums}
    # Check for duplicates first
    new_sums = set(existing_sums)
    if new_elem in new_sums:
        return False, None
        
    for s in existing_sums:
        ns = s + new_elem
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
        
    # Check 1: New sums must not be multiples of any existing sum (or vice versa, but we check both)
    # We need to check:
    # A) Does any new sum divide any old sum?
    # B) Does any old sum divide any new sum?
    # C) Do any two new sums divide each other?
    
    old_sums_list = list(existing_sums)
    new_only_sums = [s for s in new_sums if s not in existing_sums] # This is {new_elem} U {s+new_elem}
    
    # Optimization: Sort old sums to help with early termination?
    # For A: For each n in new_only_sums, check if any s in old_sums_list is divisible by n.
    # For B: For each n in new_only_sums, check if n is divisible by any s in old_sums_list.
    
    # Let's combine A and B checks against old sums.
    for n in new_only_sums:
        for s in old_sums_list:
            if n == s:
                continue # Already handled by duplicate check
            if n > s:
                if n % s == 0:
                    return False, None
            else:
                if s % n == 0:
                    return False, None
    
    # Check C: Do any two new_only_sums divide each other?
    # Note: new_only_sums contains new_elem and shifted sums.
    # We don't need to check against old sums again.
    no_sorted = sorted(new_only_sums)
    for i in range(len(no_sorted)):
        a = no_sorted[i]
        for j in range(i + 1, len(no_sorted)):
            b = no_sorted[j]
            if b % a == 0:
                return False, None
                
    return True, new_sums

def _greedy_construct(pool, rng, max_size=30):
    """Greedy construction with random order."""
    result = []
    result_sums = set()
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        valid, new_sums = _can_add_fast(result, elem, result_sums)
        if valid:
            result.append(elem)
            result_sums = new_sums
            if len(result) >= max_size:
                break
    return result

def _local_search_improve(nums, pool, rng, max_rounds=20):
    """Improve a set by trying swaps and additions."""
    current = list(nums)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        improved = False
        
        # Try adding elements
        rng.shuffle(available)
        for elem in available:
            valid, new_sums = _can_add_fast(current, elem, _compute_all_sums(current))
            if valid:
                current.append(elem)
                current_set.add(elem)
                available.remove(elem)
                improved = True
                break
        
        # Try swaps
        if not improved and available and current:
            # Try to swap an element in current with one in available
            # We want to increase size, so swaps are only useful if they allow a subsequent add?
            # Or if we are stuck at a local maximum.
            # Let's try random swaps to escape local minima.
            for i in range(len(current)):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                old_elem = current[i]
                for new_elem in available:
                    trial = current[:i] + current[i+1:] + [new_elem]
                    if _is_valid_set(trial):
                        # If valid, we might be able to add more later.
                        # Accept if it doesn't reduce size (it doesn't) and maybe if it looks "better" (heuristic)?
                        # For now, just accept to explore.
                        current = trial
                        current_set = set(current)
                        available.remove(new_elem)
                        available.append(old_elem)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return current

def _compute_all_sums(nums):
    """Generate all nonempty subset sums of nums as a set."""
    sums = set()
    n = len(nums)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Strategy 1: Multiple greedy passes with different seeds, focusing on large numbers
    # Large numbers have fewer multiples, making the constraint easier to satisfy.
    large_pool = list(range(128, 257))
    
    for seed in range(50):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        rng = random.Random(seed)
        # Mix large and medium numbers
        current_pool = large_pool.copy()
        if seed % 2 == 0:
            current_pool += list(range(64, 128))
        
        result = _greedy_construct(current_pool, rng)
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Strategy 2: Local search improvement on best found
    if best and best_size < 30:
        if time.time() - START_TIME < TIME_LIMIT - 10:
            rng = random.Random(999)
            improved = _local_search_improve(best, pool, rng, max_rounds=20)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Strategy 3: Try to build from scratch with a focus on spread-out values
    if time.time() - START_TIME < TIME_LIMIT - 5:
        # Use values that are far apart
        spread_pool = [x for x in range(128, 257) if x % 2 != 0] # Odd numbers
        rng = random.Random(4000)
        result = _greedy_construct(spread_pool, rng)
        if result and len(result) < 30:
            result = _local_search_improve(result, pool, rng, max_rounds=10)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    # Final fallback
    if not best:
        best = [31, 32, 34]
    
    # Ensure Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
