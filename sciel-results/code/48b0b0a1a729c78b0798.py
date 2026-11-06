import time
from fractions import Fraction

_START_TIME = None
_TIME_LIMIT = 85.0  # 90s limit, keep 5s buffer

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _subset_sum_possible(target, items):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of 'items' (list of Fractions).
    items are all positive.
    Returns True if possible, False otherwise.
    """
    if target == 0:
        return True
    if not items:
        return False
    if target < 0:
        return False
    
    # Sort items descending to hit large sums quickly and prune
    items_sorted = sorted(items, key=lambda x: (x.numerator, -x.denominator), reverse=True)
    
    n = len(items_sorted)
    # Precompute suffix sums for lower bound pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]
        
    # If total sum < target, impossible
    if suffix_sums[0] < target:
        return False
        
    # If the smallest item is larger than target, impossible
    if items_sorted[-1] > target:
        return False

    # Memoization
    global _SUBSET_SUM_CACHE
    items_key = tuple((item.numerator, item.denominator) for item in items_sorted)
    target_key = (target.numerator, target.denominator)
    
    cache_key = (target_key, items_key)
    if cache_key in _SUBSET_SUM_CACHE:
        return _SUBSET_SUM_CACHE[cache_key]
        
    def dfs(idx, current_sum):
        if _time_exceeded():
            return False
        if current_sum == target:
            return True
        if idx == n:
            return False
        if current_sum > target:
            return False
        # Prune if even taking all remaining items can't reach target
        if current_sum + suffix_sums[idx] < target:
            return False
            
        # Try taking items[idx]
        if dfs(idx + 1, current_sum + items_sorted[idx]):
            return True
        # Try skipping items[idx]
        if dfs(idx + 1, current_sum):
            return True
        return False
        
    result = dfs(0, Fraction(0))
    _SUBSET_SUM_CACHE[cache_key] = result
    return result

_SUBSET_SUM_CACHE = {}

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    """
    if x in S:
        return False
        
    recip_x = Fraction(1, x)
    recips_S = [Fraction(1, y) for y in S]
    
    # Check 1: 1/x is not a subset sum of recips_S
    if _subset_sum_possible(recip_x, recips_S):
        return False
        
    # Check 2: For each y in S, 1/y is not a subset sum of S ∪ {x} \ {y}
    for y in S:
        if y < x:
            # 1/y > 1/x, so target is positive
            target = Fraction(1, y) - recip_x
            # Items are reciprocals of S excluding y and x
            items = [Fraction(1, z) for z in S if z != y and z != x]
            if _subset_sum_possible(target, items):
                return False
        # If y > x, target is negative, safe.
        
    return True

def _greedy_initial():
    """
    Greedy strategy: Add elements from 24 down to 1.
    """
    S = []
    for x in range(24, 0, -1):
        if _can_add(x, S):
            S.append(x)
    return S

def _local_search(S, max_iters=10000):
    """
    Local search: try to add elements, and try swap moves (remove one, add two).
    """
    S = list(S)
    improved = True
    iters = 0
    while improved and iters < max_iters and not _time_exceeded():
        improved = False
        iters += 1
        
        # Try adding elements
        for x in range(24, 0, -1):
            if x not in S:
                if _can_add(x, S):
                    S.append(x)
                    improved = True
                    break  # Restart loop to re-evaluate
        
        if improved:
            continue
            
        # Try swap moves: remove one, add two
        # We want to increase size, so we remove 1 and add 2 (net +1)
        # Or remove 0 and add 1 (handled above)
        # Or remove 1 and add 1 (size same, might help future adds? No, we want max size)
        # Let's stick to net +1 moves.
        
        # Try removing one element y and adding two elements x1, x2
        for y in list(S):
            S_temp = [z for z in S if z != y]
            # Try adding pairs to S_temp
            for x1 in range(24, 0, -1):
                if x1 in S_temp or x1 == y:
                    continue
                if _can_add(x1, S_temp):
                    S_temp_new = S_temp + [x1]
                    for x2 in range(24, 0, -1):
                        if x2 in S_temp_new or x2 == y or x2 == x1:
                            continue
                        if _can_add(x2, S_temp_new):
                            # Found a swap that increases size
                            S = S_temp_new + [x2]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
                
    return S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    _SUBSET_SUM_CACHE.clear()
    
    # 1. Get a good initial solution
    best_set = _greedy_initial()
    
    # 2. Local search to improve
    best_set = _local_search(best_set)
    
    # 3. Branch and Bound to find the true maximum
    # We use a standard B&B but with a strong upper bound:
    # Upper bound = len(S) + (number of remaining elements that can be added independently?)
    # Actually, the simple bound len(S) + (24 - i) is weak.
    # A better bound: len(S) + (number of elements in [i+1, 24] that are compatible with S)
    # But compatibility is pairwise, not independent.
    
    # Let's stick to the simple B&B but with the greedy order.
    # Since N=24 is small, and we have a good initial solution,
    # the B&B should prune a lot.
    
    def search(i, S):
        nonlocal best_set
        if _time_exceeded():
            return
            
        # Pruning: Even if we add all remaining elements, we can't beat best_set
        if len(S) + (24 - i) <= len(best_set):
            return
            
        if i == 24:
            if len(S) > len(best_set):
                best_set = list(S)
            return
            
        x = i + 1
        
        # Strategy: Exclude first for small numbers (they are restrictive)
        search(i + 1, S)
        
        # Strategy 2: Include x
        if _can_add(x, S):
            S.append(x)
            search(i + 1, S)
            S.pop()
            
    search(0, [])
    
    if not best_set:
        return [24]
        
    return best_set
