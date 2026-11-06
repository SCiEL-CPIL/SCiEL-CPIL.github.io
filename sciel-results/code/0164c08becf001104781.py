import time
import math
import random

_START = time.time()
_DEADLINE = 55.0  # Safety margin

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

def _is_subset_sum(target, items):
    if target < 0:
        return False
    if target == 0:
        return True
    n = len(items)
    if n == 0:
        return False
    
    # Quick sum check
    total = sum(items)
    if total < target:
        return False
    
    # Filter out items > target (can't be part of any valid subset sum = target)
    filtered = [x for x in items if x <= target]
    if not filtered:
        return False
    if sum(filtered) < target:
        return False
    
    n = len(filtered)
    
    # For small n, use bitmask DP
    if n <= 20:
        # Bitmask DP: reachable sums
        reachable = {0}
        for item in filtered:
            new_reachable = set()
            for s in reachable:
                ns = s + item
                if ns <= target:
                    new_reachable.add(ns)
            reachable |= new_reachable
            if target in reachable:
                return True
        return target in reachable
    else:
        # Meet in the middle for larger n
        half = n // 2
        left_items = filtered[:half]
        right_items = filtered[half:]
        
        left_sums = {0}
        for item in left_items:
            new_sums = set()
            for s in left_sums:
                ns = s + item
                if ns <= target:
                    new_sums.add(ns)
            left_sums |= new_sums
            
        right_sums = {0}
        for item in right_items:
            new_sums = set()
            for s in right_sums:
                ns = s + item
                if ns <= target:
                    new_sums.add(ns)
            right_sums |= new_sums
            
        # Check intersection
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
    target_weight = _W[elem - 1]
    other_weights = [_W[x - 1] for x in candidate_set]
    
    if _is_subset_sum(target_weight, other_weights):
        return False
    
    for x in candidate_set:
        target_w = _W[x - 1]
        others = [y for y in candidate_set if y != x]
        others.append(elem)
        other_w = [_W[y - 1] for y in others]
        if _is_subset_sum(target_w, other_w):
            return False
            
    return True

def _is_fully_valid(candidate_set):
    n = len(candidate_set)
    for i, x in enumerate(candidate_set):
        target_w = _W[x - 1]
        others = [y for y in candidate_set if y != x]
        other_w = [_W[y - 1] for y in others]
        if _is_subset_sum(target_w, other_w):
            return False
    return True

def _greedy_descending():
    result = []
    for elem in range(24, 0, -1):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def _try_add_all(current_set, elem):
    """Try to add elem to current_set, return True if valid."""
    return _is_valid_addition(list(current_set), elem)

def _local_search(current_set, time_limit=2.0):
    """Local search to improve a given set. Returns improved set."""
    current = set(current_set)
    best = set(current)
    best_size = len(current)
    start_time = time.time()
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        if not current:
            break
        
        # Try to add a random element
        available = [x for x in range(1, 25) if x not in current]
        if available:
            elem = random.choice(available)
            if _is_valid_addition(list(current), elem):
                current.add(elem)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                    no_improve_count = 0
                    continue
        
        # Try a swap
        if len(current) > 1 and available:
            removed = random.choice(list(current))
            added = random.choice(available)
            new_set = (current - {removed}) | {added}
            if _is_fully_valid(list(new_set)):
                current = new_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                    no_improve_count = 0
                    continue
        
        # Occasionally remove an element to escape local optima
        if len(current) > 3 and random.random() < 0.1:
            elem = random.choice(list(current))
            current.remove(elem)
        
        no_improve_count += 1
    
    return best

def construct():
    random.seed(42)
    
    # Start with greedy descending
    initial = _greedy_descending()
    best_set = list(initial)
    best_size = len(best_set)
    
    # Multiple restarts with different random seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        if _time_left() < 3.0:
            break
        
        random.seed(seed)
        
        # Start from a random subset of the greedy solution
        if len(initial) > 4:
            # Take a random subset
            k = random.randint(max(3, len(initial) - 3), len(initial))
            current = set(random.sample(initial, k))
        else:
            current = set(initial)
        
        # Local search
        improved = _local_search(current, time_limit=3.0)
        
        if len(improved) > best_size:
            best_set = list(improved)
            best_size = len(best_set)
    
    # Final polish: try to add elements to best_set
    if _time_left() > 2.0:
        current = set(best_set)
        for elem in range(24, 0, -1):
            if elem not in current:
                if _is_valid_addition(list(current), elem):
                    current.add(elem)
        if len(current) > best_size:
            best_set = list(current)
            best_size = len(best_set)
    
    return best_set
