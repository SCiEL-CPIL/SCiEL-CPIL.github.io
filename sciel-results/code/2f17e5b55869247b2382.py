import time
import math
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = _L * i // math.gcd(_L, i)

# Precompute values V[k] = L/k
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _L // i

# Cache for subset-sum feasibility
# Key: (target, tuple(sorted(items)))
# We can use a simpler key: since items are always a subset of a fixed set,
# we can represent the "available" set as a bitmask or just use the tuple.
# Given the small size, tuple is fine.
_sum_cache = {}

def _can_form(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle with caching.
    """
    if target == 0:
        return True
    if not items:
        return target == 0
    
    # Pruning: if sum(items) < target, return False
    if sum(items) < target:
        return False
        
    # Canonicalize items for caching
    key_items = tuple(sorted(items))
    key = (target, key_items)
    
    if key in _sum_cache:
        return _sum_cache[key]
        
    n = len(key_items)
    mid = n // 2
    left_items = key_items[:mid]
    right_items = key_items[mid:]
    
    # Generate subset sums for left
    left_sums = {0}
    for item in left_items:
        new_sums = {s + item for s in left_sums}
        left_sums.update(new_sums)
        
    # Check if target is in left_sums
    if target in left_sums:
        _sum_cache[key] = True
        return True
        
    # Generate subset sums for right
    right_sums = {0}
    for item in right_items:
        new_sums = {s + item for s in right_sums}
        right_sums.update(new_sums)
        
    # Check if there's a pair
    # Iterate over the smaller set for efficiency
    if len(left_sums) < len(right_sums):
        found = False
        for l in left_sums:
            if (target - l) in right_sums:
                found = True
                break
        if found:
            _sum_cache[key] = True
            return True
    else:
        found = False
        for r in right_sums:
            if (target - r) in left_sums:
                found = True
                break
        if found:
            _sum_cache[key] = True
            return True
            
    _sum_cache[key] = False
    return False

def _is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    """
    # S is a list of integers
    # We need to check for each x in S if 1/x is a subsum of 1/others
    # Using integers: V[x] is a subsum of {V[y] for y in S, y != x}
    
    # Optimization: check smaller elements first? Or just iterate.
    # If we find one violation, return False immediately.
    for x in S:
        target = _V[x]
        others = [_V[y] for y in S if y != x]
        if _can_form(target, others):
            return False
    return True

def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    """
    all_elements = list(range(1, 25))
    
    # 1. Greedy Initialization
    # Try adding from large to small
    best_set = []
    for candidate in range(24, 0, -1):
        trial = best_set + [candidate]
        if _is_valid(trial):
            best_set = trial
            
    # Also try small to large?
    trial_small = []
    for candidate in range(1, 25):
        t = trial_small + [candidate]
        if _is_valid(t):
            trial_small = t
    if len(trial_small) > len(best_set):
        best_set = trial_small
        
    # Also try a few random greedy orders
    random.seed(42)
    for _ in range(20):
        order = all_elements[:]
        random.shuffle(order)
        g_set = []
        for candidate in order:
            trial = g_set + [candidate]
            if _is_valid(trial):
                g_set = trial
        if len(g_set) > len(best_set):
            best_set = g_set
            
    current_set = list(best_set)
    best_size = len(current_set)
    best_set_found = list(current_set)
    
    # 2. Local Search
    # We will use a simple local search with swaps and adds.
    # To speed up, we maintain the set as a list.
    
    # We'll run a loop until time runs out.
    # Each iteration:
    # 1. Try to add any missing element (random order).
    # 2. If no add, try a random swap.
    # 3. If no swap, try a random double-swap or restart?
    #    For now, just random swap.
    
    # To make it more effective, we can use a "temperature" or just accept
    # any valid move. Since we maximize size, any add is an improvement.
    # Swaps maintain size but might enable future adds.
    
    while _time_remaining() > 2.0:
        added = False
        missing = [x for x in all_elements if x not in current_set]
        
        if missing:
            random.shuffle(missing)
            for x in missing:
                trial = current_set + [x]
                if _is_valid(trial):
                    current_set = trial
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set_found = list(current_set)
                    break # Restart add phase if we added something
        
        if not added:
            # Try a swap
            if current_set:
                u_idx = random.randint(0, len(current_set) - 1)
                u = current_set[u_idx]
                
                if missing:
                    v = random.choice(missing)
                    new_set = list(current_set)
                    new_set[u_idx] = v
                    
                    if _is_valid(new_set):
                        current_set = new_set
                        # Note: Size is same, so best_size doesn't update here.
                        # But this might allow a future Add.
    
    return sorted(best_set_found)
