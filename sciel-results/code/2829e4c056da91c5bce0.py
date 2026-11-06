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
    if sum(items) < target:
        return False
        
    # Meet in the middle
    half = n // 2
    left_items = items[:half]
    right_items = items[half:]
    
    left_sums = {0}
    for item in left_items:
        # Optimization: if item > target, we can't add it to any sum <= target
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

def _greedy_ascending():
    result = []
    for elem in range(1, 25):
        if _is_valid_addition(result, elem):
            result.append(elem)
    return result

def _local_search_improve(current_set):
    """
    Perform best-improvement local search with 1-swaps and 2-swaps.
    """
    current = set(current_set)
    improved = True
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
                    break # Restart loop to check for more additions
        
        if added:
            continue

        # 2. Try 1-swaps: remove x, add y
        swapped = False
        for x in list(current):
            if _time_left() < 1.0:
                break
            temp_set = current - {x}
            for y in range(1, 25):
                if y in current:
                    continue
                if _is_valid_addition(list(temp_set), y):
                    current = temp_set | {y}
                    swapped = True
                    improved = True
                    break
            if swapped:
                break
        
        if swapped:
            continue

        # 3. Try 2-swaps: remove x, z, add y, w
        # This is expensive: O(N^4) checks. N=24. 24*23*24*23 ~ 300k checks.
        # Each check is O(2^(N/2)) ~ 5000 ops. Total ~ 1.5 billion ops. Too slow in Python.
        # We need to be smarter or limit the search.
        
        # Instead of full 2-swaps, try random 2-swaps if time permits?
        # Or just stick to 1-swaps but with random restarts?
        
        # Let's try a limited 2-swap search:
        # Pick a random pair to remove, try all pairs to add.
        if _time_left() > 5.0:
            # Randomized 2-swap exploration
            elems = list(current)
            if len(elems) >= 2:
                # Try a few random 2-swaps
                for _ in range(10):
                    if _time_left() < 5.0:
                        break
                    x, z = random.sample(elems, 2)
                    temp_set = current - {x, z}
                    found = False
                    # Try all pairs y, w not in current
                    avail = [i for i in range(1, 25) if i not in current]
                    for i in range(len(avail)):
                        for j in range(i+1, len(avail)):
                            y, w = avail[i], avail[j]
                            # Check if adding y, w to temp_set is valid
                            # We can check validity of adding y first, then w?
                            # No, adding y might be invalid on its own but valid with w? 
                            # Actually, if {S \ {x,z}} U {y,w} is valid, then adding y to S\{x,z} might not be valid 
                            # if y forms a sum with existing elements? 
                            # Wait, the condition is pairwise. 
                            # If S' = S \ {x,z} U {y,w} is valid, then for any element e in S', 
                            # 1/e is not sum of reciprocals of S'\{e}.
                            # This implies that adding y to S\{x,z} is NOT necessarily valid, 
                            # because y might form a sum with x or z? No, x and z are removed.
                            # But y might form a sum with other elements in S\{x,z}?
                            # If y forms a sum with elements in S\{x,z}, then S' would be invalid.
                            # So, if S' is valid, then y must be addable to S\{x,z} WITHOUT w?
                            # Let's check: 
                            # Condition for y in S': 1/y != sum_{u in S'\{y}} 1/u.
                            # S'\{y} = (S \ {x,z}) \ {w} U {} ... wait.
                            # S'\{y} = (S \ {x,z,w}).
                            # So we need 1/y != sum_{u in S \ {x,z,w}} 1/u.
                            # This is different from adding y to S\{x,z} (which checks 1/y != sum_{u in S\{x,z}} 1/u).
                            # So we cannot just use _is_valid_addition sequentially.
                            # We must check the full set validity for S'.
                            
                            new_set = temp_set | {y, w}
                            if _is_fully_valid(list(new_set)):
                                current = new_set
                                found = True
                                improved = True
                                break
                        if found:
                            break
                    if found:
                        break

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
        
    # 3. Randomized Greedy (more iterations)
    random.seed(42)
    for i in range(50):
        if _time_left() < 10.0:
            break
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
            
    # Final check to ensure validity (paranoia)
    if not _is_fully_valid(best_set):
        # Fallback to a known valid small set if something went wrong
        return [1, 2, 4, 8, 16]
        
    return best_set
