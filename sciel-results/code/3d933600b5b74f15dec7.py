import random
import time
import sys

L = 5354228880  # lcm(1..24)

# Precompute weights for each element
WEIGHTS = {i: L // i for i in range(1, 25)}

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle with early termination.
    """
    n = len(weights)
    if n == 0:
        return target == 0
    
    if target == 0:
        return True
    
    # Filter weights > target (positive weights only)
    weights = [w for w in weights if w <= target]
    if not weights:
        return target == 0
    
    # Quick check: if sum of all <= target weights < target, impossible
    total = sum(weights)
    if total < target:
        return False
    
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    # Generate all subset sums for left half, keeping only those <= target
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # Generate all subset sums for right half, keeping only those <= target
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check intersection: is there s in left_sums such that target-s in right_sums?
    # Iterate over the smaller set
    if len(left_sums) < len(right_sums):
        for s in left_sums:
            if (target - s) in right_sums:
                return True
    else:
        for s in right_sums:
            if (target - s) in left_sums:
                return True
    return False

def check_violation(S):
    """
    Check if S has a violation. Returns (violated, violating_x) or (False, None).
    S is a list/set of ints.
    """
    if not S:
        return False, None
    S_list = list(S)
    for x in S_list:
        target = WEIGHTS[x]
        weights = [WEIGHTS[y] for y in S_list if y != x]
        if subset_sum_exists(weights, target):
            return True, x
    return False, None

def is_valid(S):
    """
    Check if S is a valid set.
    S is a list of ints.
    """
    violated, _ = check_violation(S)
    return not violated

def greedy_insertion(order, deadline):
    """
    Try to build a valid set by greedily inserting elements in the given order.
    Returns the resulting set.
    """
    S = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S:
            continue
        new_S = S | {x}
        violated, _ = check_violation(new_S)
        if not violated:
            S = new_S
    return S

def local_search_swap(S, deadline):
    """
    Try to improve S by swapping elements out and adding others.
    Returns improved set.
    """
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [i for i in range(1, 25) if i not in S]
        if not missing:
            break
        random.shuffle(missing)
        for m in missing:
            if time.time() > deadline:
                break
            # Try adding m directly
            new_S = S | {m}
            violated, _ = check_violation(new_S)
            if not violated:
                S = new_S
                improved = True
                break
            # Try swapping: remove one element from S, add m
            S_list = list(S)
            random.shuffle(S_list)
            for r in S_list:
                if time.time() > deadline:
                    break
                candidate = (S - {r}) | {m}
                violated2, _ = check_violation(candidate)
                if not violated2:
                    S = candidate
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin (60s limit)
    
    best_set = None
    best_size = 0
    
    # Initial guess: known valid seed
    seed_set = list(range(13, 25))
    if is_valid(seed_set):
        best_set = seed_set
        best_size = len(seed_set)
    
    # Strategy 1: Multiple greedy insertions with different random orders
    # Reduced trials to ensure we finish well within the deadline
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S = greedy_insertion(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 2: Local search (swap) from best set
    if best_set is not None:
        S = set(best_set)
        S = local_search_swap(S, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 3: More greedy insertions with bias toward larger elements
    # Larger elements have smaller reciprocals, making violations less likely
    for trial in range(50):
        if time.time() > deadline:
            break
        # Bias: larger elements first (more likely to be valid)
        order = list(range(24, 0, -1))
        random.shuffle(order)
        # Mix: 50% chance to keep biased order, else fully random
        if random.random() < 0.5:
            order = list(range(1, 25))
            random.shuffle(order)
        S = greedy_insertion(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        # Also try local search from this result
        if len(S) >= best_size - 1:
            S2 = local_search_swap(S, deadline)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = list(S2)
    
    # Strategy 4: Start from full set, repair, then try to add back
    for trial in range(20):
        if time.time() > deadline:
            break
        S = set(range(1, 25))
        # Repair: remove violating elements
        while time.time() < deadline:
            violated, x = check_violation(S)
            if not violated or not S:
                break
            S.discard(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Try to add back missing elements
        missing = [i for i in range(1, 25) if i not in S]
        random.shuffle(missing)
        for m in missing:
            if time.time() > deadline:
                break
            new_S = S | {m}
            violated2, _ = check_violation(new_S)
            if not violated2:
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
        
        # Local search
        S = local_search_swap(S, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        return [1]
        
    return best_set
