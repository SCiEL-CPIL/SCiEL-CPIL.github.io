import random
import time

L = 5354228880  # lcm(1..24)
L_weights = [0] + [L // i for i in range(1, 25)]

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle with early termination.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
    if sum(weights) < target:
        return False
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        if target in left_sums:
            return True
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        left_sums.update(new_sums)
        
    right_sums = {0}
    for w in right:
        if target in right_sums:
            return True
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def is_valid(S):
    """
    Check if S is a valid set.
    S is a list of ints.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = L_weights[x]
        weights = [L_weights[y] for y in S_set if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def find_violation(S):
    """
    Find a violation in S. Returns (x, contributing_subset) or None if valid.
    """
    S_set = set(S)
    for x in S:
        target = L_weights[x]
        weights = [(L_weights[y], y) for y in S_set if y != x]
        weights = [(w, y) for w, y in weights if w <= target]
        if not weights:
            continue
        if sum(w for w, _ in weights) < target:
            continue
        
        n = len(weights)
        mid = n // 2
        left = weights[:mid]
        right = weights[mid:]
        
        # Map sum to (sum, set of elements)
        left_sums = {0: frozenset()}
        for w, y in left:
            new_sums = {}
            for s, elems in left_sums.items():
                ns = s + w
                if ns <= target:
                    if ns not in left_sums or len(elems) < len(left_sums[ns]):
                        left_sums[ns] = elems | {y}
            for s, elems in list(new_sums.items()):
                pass # Not needed for logic, just update left_sums in place carefully?
            # Actually, standard MITM with sets of elements is tricky to update in place without copying.
            # Let's just use the boolean check first, and if found, we don't strictly need the subset for removal heuristic
            # unless we want to remove a specific contributor.
            # For now, let's just return x.
            pass
        
        # Simplified: Just check existence
        if subset_sum_exists([w for w, _ in weights], target):
            return x
    return None

def can_add(S, x):
    """
    Check if x can be added to valid set S.
    Returns True if S U {x} is valid.
    """
    if x in S:
        return False
    
    S_set = set(S)
    
    # 1. Check if x is covered by S
    target_x = L_weights[x]
    weights_for_x = [L_weights[y] for y in S_set]
    if subset_sum_exists(weights_for_x, target_x):
        return False
        
    # 2. Check if adding x covers any y in S
    for y in S:
        target_diff = L_weights[y] - L_weights[x]
        if target_diff < 0:
            continue
        if target_diff == 0:
            continue
            
        weights_for_y = [L_weights[z] for z in S_set if z != y and z != x]
        if subset_sum_exists(weights_for_y, target_diff):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best_set = None
    best_size = 0
    
    # Initial guess: Seed solution
    seed_set = list(range(13, 25))
    if is_valid(seed_set):
        best_set = seed_set
        best_size = len(seed_set)
    
    # Deterministic greedy construction
    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    # Try several deterministic orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search
    def local_search(S, max_iterations=100):
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each missing element
            missing = [i for i in range(1, 25) if i not in S_set]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    return S
                if can_add(S, m):
                    S.append(m)
                    S_set.add(m)
                    improved = True
            
            # Try 2-for-1 swap: remove one, add two
            if not improved and len(S) >= 1:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    y = S[i]
                    S_without_y = [x for x in S if x != y]
                    S_without_y_set = set(S_without_y)
                    candidates = [k for k in range(1, 25) if k not in S_without_y_set and k != y]
                    random.shuffle(candidates)
                    
                    for c1 in candidates:
                        if time.time() > deadline:
                            return S
                        if can_add(S_without_y, c1):
                            S_temp = S_without_y + [c1]
                            S_temp_set = S_without_y_set | {c1}
                            for c2 in candidates:
                                if c2 == c1:
                                    continue
                                if time.time() > deadline:
                                    return S
                                if can_add(S_temp, c2):
                                    S = S_temp + [c2]
                                    S_set = S_temp_set | {c2}
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
        return S

    # Apply local search to best set found so far
    if best_set is not None and time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Random restarts with repair strategy (from Inspiration 1)
    random.seed(42)
    attempts = 0
    while time.time() < deadline and best_size < 24:
        attempts += 1
        
        # Strategy 1: Start with full set and repair
        S = set(range(1, 25))
        
        while True:
            if time.time() > deadline:
                break
            if not S:
                break
                
            if is_valid(list(S)):
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
                break
            else:
                # Find a violation to guide removal
                violated_elem = find_violation(list(S))
                if violated_elem is not None:
                    S.remove(violated_elem)
                else:
                    break
                
        # After repairing to a valid set, try to add elements back
        if best_set is not None and len(S) < 24:
            missing = [i for i in range(1, 25) if i not in S]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    break
                new_S = S.copy()
                new_S.add(m)
                if is_valid(list(new_S)):
                    S = new_S
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = list(S)
            
            # Local search on the repaired set
            if time.time() < deadline:
                improved_set = local_search(list(S), max_iterations=50)
                if len(improved_set) > best_size:
                    best_size = len(improved_set)
                    best_set = improved_set

        # Strategy 2: Random restarts with greedy
        if time.time() < deadline:
            random_order = list(range(1, 25))
            random.shuffle(random_order)
            S = greedy_construct(random_order)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            
            if time.time() < deadline:
                improved_set = local_search(S, max_iterations=50)
                if len(improved_set) > best_size:
                    best_size = len(improved_set)
                    best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
        
    return best_set
