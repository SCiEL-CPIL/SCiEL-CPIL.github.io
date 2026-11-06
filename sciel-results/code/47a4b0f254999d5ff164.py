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
    
    # Early exit: if sum of all items < target, impossible
    # We can pre-compute sum of items?
    # items are positive.
    
    # Use meet in the middle
    half = n // 2
    left_items = items[:half]
    right_items = items[half:]
    
    left_sums = {0}
    for item in left_items:
        # Optimization: if item > target, we can't add it to any sum <= target
        # But we need sums <= target.
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
    # Iterate over smaller set
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

def _greedy_ascending():
    result = []
    for elem in range(1, 25):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def _local_search_improve(current_set):
    """
    Perform deterministic local search.
    Try to add any element.
    Try to swap any element with any other.
    If no addition found, try random valid swaps to explore the space.
    """
    current = set(current_set)
    improved = True
    stall_count = 0
    max_stall = 20 # Allow some exploration
    
    while improved:
        if _time_left() < 1.0:
            break
        improved = False
        
        # 1. Try adding any element not in current
        added = False
        for elem in range(1, 25):
            if elem not in current:
                if _is_valid_addition(list(current), elem):
                    current.add(elem)
                    added = True
                    improved = True
                    stall_count = 0
                    break # Restart loop to check for more additions
        
        if added:
            continue

        # If no addition was found, we are stuck at this size.
        # Try to escape local optimum by swapping.
        
        # 2. Try deterministic swaps first (remove x, add y)
        swapped = False
        for x in list(current):
            if _time_left() < 1.0:
                break
            temp_set = current - {x}
            for y in range(1, 25):
                if y in current:
                    continue
                if y == x:
                    continue
                if _is_valid_addition(list(temp_set), y):
                    current = temp_set | {y}
                    swapped = True
                    improved = True
                    stall_count = 0
                    break
            if swapped:
                break
                
        if swapped:
            continue
        
        # 3. If no deterministic swap worked, try random valid swaps
        # This explores the space of valid sets of the same size.
        random_swapped = False
        for _ in range(10): # Try a few random swaps
            if _time_left() < 1.0:
                break
            if not current:
                break
            x = random.choice(list(current))
            candidates_for_swap = [y for y in range(1, 25) if y not in current and y != x]
            if not candidates_for_swap:
                break
            y = random.choice(candidates_for_swap)
            temp_set = current - {x}
            if _is_valid_addition(list(temp_set), y):
                current = temp_set | {y}
                random_swapped = True
                improved = True
                stall_count += 1
                break
        
        if random_swapped:
            if stall_count > max_stall:
                # Too much wandering without improvement?
                # Actually, we just want to keep exploring.
                # But if we've done many swaps, maybe stop?
                # Let's just let it run until time runs out or we find an addition.
                pass
            continue
        
        # If no addition, no deterministic swap, no random swap worked
        improved = False

    return list(current)

def construct():
    # Try a few different starting points
    best_set = []
    best_size = 0
    
    # 1. Greedy Descending
    s1 = _greedy_descending()
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # 2. Greedy Ascending
    s2 = _greedy_ascending()
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # 3. Randomized Greedy (a few times)
    random.seed(42)
    for _ in range(10):
        elems = list(range(1, 25))
        random.shuffle(elems)
        s_rand = []
        for elem in elems:
            if _is_valid_addition(s_rand, elem):
                s_rand.append(elem)
        if len(s_rand) > best_size:
            best_set = s_rand
            best_size = len(s_rand)
            
    # 4. Local Search from the best starting point
    # Also try local search from other good starting points if time permits
    candidates = [best_set]
    if s1 != best_set: candidates.append(s1)
    if s2 != best_set: candidates.append(s2)
    
    for start in candidates:
        if _time_left() < 5.0:
            break
        improved_set = _local_search_improve(start)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # 5. Try more random restarts of local search if time permits
    # The local search now includes random swaps, so it's more powerful.
    # We can just run it on the best set found.
    if _time_left() > 5.0:
        # Run local search again on the best set, maybe with different random seed?
        # The function uses random, so it will be different.
        improved_set = _local_search_improve(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    # Final check to ensure validity (paranoia)
    if not _is_fully_valid(best_set):
        # Fallback to a known valid small set if something went wrong
        return [1, 2, 4, 8, 16]
        
    return best_set
