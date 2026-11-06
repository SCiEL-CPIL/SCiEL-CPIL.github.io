import time
import math
import random

_START = time.time()
_DEADLINE = 85.0  # Safety margin under 90 s wall-clock

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

# Precompute weights
_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

def _is_subset_sum(target, items):
    """Check if target can be formed by a subset sum of items.
    Uses meet-in-the-middle for efficiency."""
    if target < 0:
        return False
    if target == 0:
        return True  # Empty subset sums to 0

    n = len(items)
    if n == 0:
        return target == 0

    # If target is larger than sum of all items, impossible
    total = sum(items)
    if total < target:
        return False

    # If target equals total, it's possible (use all items)
    if total == target:
        return True

    # Meet in the middle
    half = n // 2
    left_items = items[:half]
    right_items = items[half:]

    # Generate all subset sums of left_items that are <= target
    left_sums = {0}
    for item in left_items:
        new_sums = set()
        for s in left_sums:
            ns = s + item
            if ns <= target:
                new_sums.add(ns)
        left_sums |= new_sums

    # Generate all subset sums of right_items that are <= target
    right_sums = {0}
    for item in right_items:
        new_sums = set()
        for s in right_sums:
            ns = s + item
            if ns <= target:
                new_sums.add(ns)
        right_sums |= new_sums

    # Check if any combination of left and right sums equals target
    if len(left_sums) < len(right_sums):
        for ls in left_sums:
            if (target - ls) in right_sums:
                return True
    else:
        for rs in right_sums:
            if (target - rs) in left_sums:
                return True
    return False

def _is_valid_addition(candidate_set, elem):
    """Check if adding elem to candidate_set (list of 1-based elements) is valid.
    Uses incremental check: only check new constraints introduced by elem."""
    target_weight = _W[elem - 1]
    other_weights = [_W[x - 1] for x in candidate_set]
    
    # 1. Check if elem is subsumed by existing elements
    if _is_subset_sum(target_weight, other_weights):
        return False
    
    # 2. Check if any existing element becomes subsumed by adding elem
    #    For each x in candidate_set, check if w[x] is a subset sum of (candidate_set \ {x}) U {elem}
    #    This means w[x] = sum(subset of candidate_set \ {x}) + (0 or 1)*w[elem]
    #    Case A: w[elem] not used. Then w[x] must be subset sum of candidate_set \ {x}.
    #            This was already false when x was added (assuming we maintain validity).
    #            But adding other elements later might have made it true, so we need to check.
    #            However, if the set was valid before adding elem, then w[x] was NOT a subset sum
    #            of candidate_set \ {x}. So Case A is already satisfied.
    #    Case B: w[elem] IS used. Then w[x] - w[elem] must be a subset sum of candidate_set \ {x}.
    for x in candidate_set:
        target_w = _W[x - 1]
        diff = target_w - target_weight
        if diff >= 0:
            others = [y for y in candidate_set if y != x]
            if others:
                other_w = [_W[y - 1] for y in others]
                if _is_subset_sum(diff, other_w):
                    return False
            else:
                if diff == 0:
                    return False
        # If diff < 0, impossible since weights are positive.
        
    return True

def _is_fully_valid(candidate_set):
    """Check if the entire set is valid."""
    n = len(candidate_set)
    for i, x in enumerate(candidate_set):
        target_w = _W[x - 1]
        others = [y for y in candidate_set if y != x]
        if not others:
            continue
        other_w = [_W[y - 1] for y in others]
        if _is_subset_sum(target_w, other_w):
            return False
    return True

def _greedy_descending():
    """Greedy construction: try elements from 24 down to 1."""
    result = []
    for elem in range(24, 0, -1):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def _greedy_ascending():
    """Greedy construction: try elements from 1 up to 24."""
    result = []
    for elem in range(1, 25):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def _local_search(initial_set, time_budget=5.0):
    """Local search to improve the set."""
    current = set(initial_set)
    best_set = list(current)
    best_size = len(current)
    start_time = time.time()
    
    while _time_left() > time_budget + 1.0 and (time.time() - start_time) < time_budget:
        if not current:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an element
            available = [x for x in range(1, 25) if x not in current]
            if not available:
                continue
            elem = random.choice(available)
            if _is_valid_addition(list(current), elem):
                current.add(elem)
                if len(current) > best_size:
                    best_set = list(current)
                    best_size = len(best_set)
        elif move_type < 0.7:
            # Try to remove an element
            if len(current) > 1:
                elem = random.choice(list(current))
                current.remove(elem)
        else:
            # Try a swap: remove one, add another
            if len(current) > 0:
                removed = random.choice(list(current))
                available = [x for x in range(1, 25) if x not in current]
                if available:
                    added = random.choice(available)
                    new_set = (current - {removed}) | {added}
                    if _is_fully_valid(list(new_set)):
                        current = new_set
                        if len(current) > best_size:
                            best_set = list(current)
                            best_size = len(best_set)
                            
    return best_set

def construct():
    random.seed(42)
    
    # Try multiple greedy strategies
    greedy_desc = _greedy_descending()
    greedy_asc = _greedy_ascending()
    
    # Pick the better greedy result
    if len(greedy_desc) >= len(greedy_asc):
        initial = greedy_desc
    else:
        initial = greedy_asc
    
    best_set = list(initial)
    best_size = len(best_set)
    
    # Local search to improve
    improved = _local_search(initial, time_budget=5.0)
    if len(improved) > best_size:
        best_set = improved
        best_size = len(best_set)
    
    # If we have time, try more local search from different starting points
    if _time_left() > 10.0:
        # Try local search from the other greedy result
        other_initial = greedy_asc if initial == greedy_desc else greedy_desc
        if other_initial != initial:
            improved2 = _local_search(other_initial, time_budget=3.0)
            if len(improved2) > best_size:
                best_set = improved2
                best_size = len(best_set)
        
        # Try random restarts
        while _time_left() > 5.0:
            random_set = random.sample(range(1, 25), random.randint(5, 12))
            # Filter to only valid sets
            valid_set = []
            for elem in sorted(random_set, reverse=True):
                if _is_valid_addition(valid_set, elem):
                    valid_set.append(elem)
            if valid_set:
                improved3 = _local_search(valid_set, time_budget=2.0)
                if len(improved3) > best_size:
                    best_set = improved3
                    best_size = len(best_size)
    
    return best_set
