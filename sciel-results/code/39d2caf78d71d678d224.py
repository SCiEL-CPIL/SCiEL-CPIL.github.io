import random
import time
import sys
from bisect import bisect_left

L = 5354228880  # lcm(1..24)

# Precompute L//x for x in 1..24
L_DIV = [0] + [L // i for i in range(1, 25)]

# Precompute all possible subset sums for each size of weight set
# This is too expensive for all sizes, so we'll compute on demand with caching

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle with sorted lists and binary search.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
    if len(weights) == 1:
        return weights[0] == target
    if len(weights) == 2:
        return weights[0] == target or weights[1] == target or weights[0] + weights[1] == target
    
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    # Generate all subset sums for left half
    left_sums = [0]
    for w in left:
        current = list(left_sums)
        for s in current:
            ns = s + w
            if ns <= target:
                left_sums.append(ns)
    left_sums.sort()
    
    # Generate all subset sums for right half, checking against left
    right_sums = [0]
    for w in right:
        current = list(right_sums)
        for s in current:
            ns = s + w
            if ns <= target:
                right_sums.append(ns)
                # Check if target - ns is in left_sums
                idx = bisect_left(left_sums, target - ns)
                if idx < len(left_sums) and left_sums[idx] == target - ns:
                    return True
        right_sums.sort()
    
    # Final check for all right_sums
    for rs in right_sums:
        idx = bisect_left(left_sums, target - rs)
        if idx < len(left_sums) and left_sums[idx] == target - rs:
            return True
    return False

# Cache for subset sum results
_subset_sum_cache = {}

def subset_sum_cached(weights_tuple, target):
    """Cached version of subset_sum_exists."""
    if target == 0:
        return True
    if not weights_tuple:
        return False
    key = (weights_tuple, target)
    if key in _subset_sum_cache:
        return _subset_sum_cache[key]
    result = subset_sum_exists(list(weights_tuple), target)
    _subset_sum_cache[key] = result
    return result

def is_valid(S):
    """
    Check if S is a valid set.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = L_DIV[x]
        weights = tuple(L_DIV[y] for y in S_set if y != x)
        if subset_sum_cached(weights, target):
            return False
    return True

def can_add(S_set, x):
    """
    Check if x can be added to valid set S_set.
    """
    # Check for x: L//x should not be formable by subset of {L//y for y in S_set}
    if S_set:
        weights_for_x = tuple(L_DIV[y] for y in S_set)
        if subset_sum_cached(weights_for_x, L_DIV[x]):
            return False
    
    # Check for each y in S_set: L//y should not be formable by subset of
    # {L//z for z in S_set, z != y} union {L//x}
    for y in S_set:
        A = tuple(L_DIV[z] for z in S_set if z != y and z != x)
        tx = L_DIV[x]
        ty = L_DIV[y]
        
        # Case 1: don't use L//x
        if A and subset_sum_cached(A, ty):
            return False
        # Case 2: use L//x
        if ty >= tx:
            if subset_sum_cached(A, ty - tx):
                return False
    
    return True

def backtracking_search(deadline, current_best_size):
    """
    Backtracking search with forward checking to find a valid set of maximum size.
    Tries to build a set by adding elements one at a time, pruning when conflicts are detected.
    """
    best_set = []
    best_size = 0
    elements = list(range(1, 25))
    
    def backtrack(S, S_set, remaining, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        if not remaining:
            return
        
        # Forward checking: for each candidate in remaining, check if it can be added
        # Try candidates in order of "least constraining" - those that conflict with fewest existing elements
        candidates = []
        for x in remaining:
            if can_add(S_set, x):
                # Count how many other remaining elements would be blocked
                blocked = 0
                for y in remaining:
                    if y != x and y not in S_set:
                        # Check if y can be added to S ∪ {x}
                        new_S_set = S_set | {x}
                        if not can_add(new_S_set, y):
                            blocked += 1
                candidates.append((blocked, x))
        
        if not candidates:
            return
        
        # Sort by least constraining first (fewest blocked)
        candidates.sort()
        
        for blocked, x in candidates:
            if time.time() > deadline:
                return
            
            new_S = S + [x]
            new_S_set = S_set | {x}
            new_remaining = [y for y in remaining if y != x]
            
            backtrack(new_S, new_S_set, new_remaining, depth + 1)
            
            if best_size >= 24:
                return
            
            # If we've found a set of size >= current_best_size, we can prune
            # if the maximum possible size from here is <= best_size
            if len(new_S) + len(new_remaining) <= best_size:
                continue
    
    # Start backtracking with empty set
    backtrack([], set(), elements, 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s total, 10s buffer)
    
    best_set = None
    best_size = 0
    
    # Initial guess: Seed solution
    seed_set = list(range(13, 25))
    if is_valid(seed_set):
        best_set = seed_set
        best_size = len(seed_set)
    
    # Deterministic greedy construction with multiple insertion orders
    def greedy_construct(order):
        """Build a valid set by greedily adding elements in the given order."""
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S
    
    # Try several deterministic orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        # Interleaved large/small
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
        # Pairs from middle outward
        [13, 12, 14, 11, 15, 10, 16, 9, 17, 8, 18, 7, 19, 6, 20, 5, 21, 4, 22, 3, 23, 2, 24, 1],
        # Start with largest
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Small first
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        # Even then odd, descending
        [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
        # Odd then even, descending
        [23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search: try to improve by adding elements
    def local_search(S, max_iterations=300):
        """Try to improve set S by adding elements."""
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
                if can_add(S_set, m):
                    S.append(m)
                    S_set.add(m)
                    improved = True
            
            # If no single addition works, try swap: remove one, add one
            if not improved and len(S) >= 1:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    S_without = [x for x in S if x != S[i]]
                    S_without_set = set(S_without)
                    candidates = [k for k in range(1, 25) if k not in S_without_set and k != S[i]]
                    random.shuffle(candidates)
                    for c in candidates[:20]:
                        if time.time() > deadline:
                            return S
                        if can_add(S_without_set, c):
                            S = S_without + [c]
                            S_set = set(S)
                            improved = True
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
    
    # Backtracking search with forward checking
    if time.time() < deadline and best_size < 24:
        bt_result = backtracking_search(deadline, best_size)
        if len(bt_result) > best_size:
            best_size = len(bt_result)
            best_set = bt_result
    
    # Random restarts with greedy + local search
    random.seed(42)
    while time.time() < deadline and best_size < 24:
        # Random order for greedy construction
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = greedy_construct(random_order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        
        # Local search on this set
        if time.time() < deadline:
            improved_set = local_search(S, max_iterations=100)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
        
    return best_set
