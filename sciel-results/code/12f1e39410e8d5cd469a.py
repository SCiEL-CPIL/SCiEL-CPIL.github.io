import time
import random

START_TIME = time.time()
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin


def _subset_sums_bitmask(nums):
    """Generate all nonempty subset sums using bitmask enumeration."""
    n = len(nums)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums


def _check_divisibility(sums_set):
    """Check if any two distinct sums in the set have a divisibility relation.
    Returns False if violation found, True otherwise."""
    sums_list = sorted(sums_set)
    m = len(sums_list)
    for i in range(m):
        si = sums_list[i]
        for j in range(i + 1, m):
            sj = sums_list[j]
            if sj % si == 0:
                return False
    return True


def _is_valid(nums):
    """Check if a set of numbers satisfies the constraint."""
    if not nums:
        return True
    sums = _subset_sums_bitmask(nums)
    return _check_divisibility(sums)


def _can_add(nums, new_elem):
    """Check if adding new_elem to nums keeps the set valid.
    Incremental: compute new sums from existing sums + new_elem, check divisibility."""
    if not nums:
        return True

    existing_sums = _subset_sums_bitmask(nums)
    
    # New sums = existing_sums ∪ {s + new_elem for s in existing_sums} ∪ {new_elem}
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    return _check_divisibility(new_sums)


def _greedy_construct(pool, rng):
    """Greedy construction: try adding elements in a given order."""
    result = []
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if _can_add(result, elem):
            result.append(elem)
            if len(result) >= 30:
                break
    return result


def _local_search(nums, pool, seed, max_rounds=20):
    """Improve the set via swap moves and then try to add more elements.
    
    Strategy:
    1. Try swapping each element with a random available element.
    2. After each successful swap, try to greedily add more elements.
    3. Repeat for max_rounds or until no improvement possible.
    """
    rng = random.Random(seed)
    current = list(nums)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        improved = False
        
        # Try swaps: for each position, try a few random replacements
        for i in range(len(current)):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Try up to 5 random candidates for this position
            candidates = rng.sample(available, min(5, len(available))) if available else []
            for new_elem in candidates:
                if new_elem in current_set:
                    continue
                
                trial = current[:i] + current[i+1:] + [new_elem]
                if _is_valid(trial):
                    # Accept the swap
                    current = trial
                    current_set = set(current)
                    available = [x for x in pool if x not in current_set]
                    improved = True
                    break
            
            if improved:
                break
        
        # After a successful swap (or even without), try to add more elements
        if available:
            rng.shuffle(available)
            for elem in available:
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                if _can_add(current, elem):
                    current.append(elem)
                    current_set.add(elem)
                    if len(current) >= 30:
                        break
            available = [x for x in pool if x not in current_set]
        
        if not improved and not available:
            break
    
    return current


def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    pool = list(range(1, 257))
    pool_desc = list(range(256, 0, -1))  # Larger numbers first (less divisibility risk)
    
    best = []
    best_size = 0
    
    # Multiple strategies:
    # 1. Greedy from random order
    # 2. Greedy from descending order (bias to large numbers)
    # 3. Greedy from medium-range bias
    # Each followed by local search
    
    max_attempts = 60
    
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        
        # Choose pool order based on attempt
        if attempt % 3 == 0:
            pool_order = pool  # Random order
        elif attempt % 3 == 1:
            pool_order = pool_desc  # Descending (large numbers first)
        else:
            # Medium-to-large bias: 60..256
            pool_order = list(range(60, 257))
            rng.shuffle(pool_order)
        
        # Greedy construction
        result = _greedy_construct(pool_order, rng)
        
        # Local search improvement
        if len(result) < 30 and time.time() - START_TIME < TIME_LIMIT - 3:
            result = _local_search(result, pool, seed=attempt * 104729 + 7, max_rounds=15)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Final fallback: if we have a decent set, try one more aggressive local search
    if best and len(best) < 30 and time.time() - START_TIME < TIME_LIMIT - 5:
        improved = _local_search(best, pool, seed=99999, max_rounds=30)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # If we didn't find anything, return a known valid small set
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]


# EVOLVE-BLOCK-START
# (The above code is the complete program)
# EVOLVE-BLOCK-END
