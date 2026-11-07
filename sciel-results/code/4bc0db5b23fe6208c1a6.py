"""Model-and-optimize approach for divisibility-incomparable subset sums.

Strategy:
- The constraint is that no two distinct nonempty subset sums divide one another.
- This is a very strong constraint. If we have numbers a < b, then a + b must not be divisible by any other subset sum, and vice versa.
- In particular, if we have two subsets S1 and S2 with sums s1 < s2, we require s2 % s1 != 0.
- Also, all subset sums must be distinct (otherwise equal sums would lead to a division issue as described in the task).

Approach:
1. Fast incremental validity check: when adding an element, only check new subset sums.
2. Greedy construction with multiple random orders.
3. Local search via swap moves to escape local optima.
4. Return the best set found within the time budget.
"""

import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin

def compute_all_sums(nums):
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

def check_divisibility(sums_set):
    """Check if any two distinct sums in the set have a divisibility relation."""
    sums_list = sorted(sums_set)
    for i in range(len(sums_list)):
        si = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % si == 0:
                return False
    return True

def is_valid_set(nums):
    """Check if a set of numbers satisfies the constraint."""
    if not nums:
        return True
    sums = compute_all_sums(nums)
    return check_divisibility(sums)

def can_add(nums, new_elem):
    """Check if adding new_elem to nums keeps the set valid.
    
    Incremental: compute existing sums, then new sums (old + new_elem), 
    and check all pairs for divisibility.
    """
    if not nums:
        return True  # Single element always valid
    
    # Get existing subset sums
    existing_sums = compute_all_sums(nums)
    
    # New subset sums are: existing_sums, and (each existing sum + new_elem), and {new_elem}
    # Actually: new sums = existing_sums ∪ {s + new_elem for s in existing_sums} ∪ {new_elem}
    # But we need to check all pairs in the combined set.
    
    # Build the full new sum set
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check divisibility in new_sums
    return check_divisibility(new_sums)

def greedy_construct(pool, order_seed=0):
    """Greedy construction: try adding elements in a given order."""
    rng = random.Random(order_seed)
    result = []
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if can_add(result, elem):
            result.append(elem)
            if len(result) >= 30:
                break
        if time.time() - START_TIME > TIME_LIMIT:
            break
    return result

def local_search_improve(nums, pool, seed=0, max_swaps=50):
    """Try to improve the set by swapping elements.
    
    For each element in the set, try replacing it with an element not in the set.
    If the swap maintains validity, accept it. Then try to add more elements.
    """
    rng = random.Random(seed)
    current = list(nums)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    
    improved = True
    swap_count = 0
    
    while improved and swap_count < max_swaps:
        improved = False
        swap_count += 1
        
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try swaps: remove one element, add another
        for i, old_elem in enumerate(current):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            for new_elem in available:
                if new_elem in current_set:
                    continue
                
                # Try the swap
                trial = current[:i] + current[i+1:] + [new_elem]
                if is_valid_set(trial):
                    # Accept the swap
                    current = trial
                    current_set = set(current)
                    improved = True
                    break
            
            if improved:
                break
        
        # After a successful swap, try to add more elements
        if improved:
            available = [x for x in pool if x not in current_set]
            rng.shuffle(available)
            for elem in available:
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                if can_add(current, elem):
                    current.append(elem)
                    current_set.add(elem)
                    if len(current) >= 30:
                        break
            available = [x for x in pool if x not in current_set]
    
    return current

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    # Pool: all integers in 1..256
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Try multiple random orders with local search improvement
    max_attempts = 40
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Greedy with random order
        result = greedy_construct(pool, order_seed=attempt)
        
        # Local search to improve
        if len(result) < 30 and time.time() - START_TIME < TIME_LIMIT - 5:
            result = local_search_improve(result, pool, seed=attempt, max_swaps=30)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
        
        # Occasionally try a different strategy: start with larger numbers
        if attempt % 4 == 0:
            pool_desc = list(range(256, 0, -1))
            result2 = greedy_construct(pool_desc, order_seed=attempt + 1000)
            if len(result2) < 30 and time.time() - START_TIME < TIME_LIMIT - 5:
                result2 = local_search_improve(result2, pool, seed=attempt + 500, max_swaps=30)
            if len(result2) > best_size:
                best = result2
                best_size = len(result2)
                if best_size >= 30:
                    break
        
        # Try medium-range bias
        if attempt % 7 == 0:
            pool_med = list(range(50, 200))
            result3 = greedy_construct(pool_med, order_seed=attempt + 2000)
            if len(result3) < 30 and time.time() - START_TIME < TIME_LIMIT - 5:
                result3 = local_search_improve(result3, pool, seed=attempt + 1500, max_swaps=30)
            if len(result3) > best_size:
                best = result3
                best_size = len(result3)
                if best_size >= 30:
                    break
    
    # If we didn't find anything good, return a known valid small set
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
