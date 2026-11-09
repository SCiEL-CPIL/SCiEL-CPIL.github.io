import time
import itertools
import random

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _can_add(subset, x):
    """Check if x can be added to subset without collision."""
    v = _LCM // x
    sums = _get_sums(subset)
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _swap_improve(subset, remove_idx, add_x, add_y):
    """Try removing element at remove_idx and adding add_x, add_y."""
    new_subset = subset[:remove_idx] + subset[remove_idx+1:] + [add_x, add_y]
    return _is_dissociated(new_subset)

def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums |= {s + v for s in sums}
    return S

def _backtracking_search(universe, target_size, time_limit, random_seed=None):
    """
    Attempt to find a dissociated set of target_size using backtracking.
    Returns a set if found, None otherwise.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    start_time = time.time()
    
    def dfs(current_set, current_sums, remaining, depth):
        nonlocal start_time
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_set) == target_size:
            return list(current_set)
            
        # Heuristic: Try candidates in a specific order
        for i, x in enumerate(remaining):
            v = _LCM // x
            is_valid = True
            for s in current_sums:
                if (s + v) in current_sums:
                    is_valid = False
                    break
            
            if is_valid:
                new_sums = current_sums | {s + v for s in current_sums}
                next_remaining = remaining[:i] + remaining[i+1:]
                result = dfs(current_set + [x], new_sums, next_remaining, depth + 1)
                if result is not None:
                    return result
                    
        return None

    # Try multiple random orders
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        order = universe.copy()
        random.shuffle(order)
        
        result = dfs([], {0}, order, 0)
        if result is not None:
            return result
            
    return None

def _local_search(best_S, universe, deadline):
    """Improve the set through local search operations."""
    best_size = len(best_S)
    improved = True
    
    while improved and time.time() < deadline - 5:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # Strategy A: Try to add a single element
        if not_in_set:
            for x in not_in_set:
                if time.time() > deadline - 3:
                    break
                if _can_add(best_S, x):
                    best_S.append(x)
                    best_size += 1
                    improved = True
                    break
        
        # Strategy B: Try 2-for-1 swaps (remove 1, add 2)
        if not improved and not_in_set and len(best_S) > 0:
            moves = []
            for idx, a in enumerate(best_S):
                for x, y in itertools.combinations(not_in_set, 2):
                    moves.append((idx, x, y))
            
            random.shuffle(moves)
            
            for idx, x, y in moves:
                if time.time() > deadline - 3:
                    break
                
                candidate = best_S[:idx] + best_S[idx+1:] + [x, y]
                
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
        
        # Strategy C: Try 1-for-1 swaps to enable future growth
        if not improved and not_in_set and len(best_S) > 0:
            moves = []
            for idx, a in enumerate(best_S):
                for x in not_in_set:
                    moves.append((idx, x))
            
            random.shuffle(moves)
            
            for idx, x in moves:
                if time.time() > deadline - 3:
                    break
                
                candidate = best_S[:idx] + [x] + best_S[idx+1:]
                
                # Check if this swap enables adding more elements
                if _is_dissociated(candidate):
                    # Try to see if we can add more elements after this swap
                    test_S = candidate.copy()
                    test_in_set = set(test_S)
                    test_not_in_set = [y for y in universe if y not in test_in_set]
                    
                    can_grow = False
                    for y in test_not_in_set:
                        if _can_add(test_S, y):
                            can_grow = True
                            break
                    
                    if can_grow:
                        best_S = candidate
                        improved = True
                        break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    best_S = []
    best_size = 0
    
    # 1. Try multiple greedy orderings to find a strong initial set
    orders = [
        sorted(universe, reverse=True),  # Largest first
        sorted(universe),                 # Smallest first
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1, 12],
    ]
    
    # Random orders
    for _ in range(30):
        random_order = universe.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # 2. Local search to improve
    if time.time() < deadline - 20:
        best_S = _local_search(best_S, universe, deadline)
        best_size = len(best_S)
    
    # 3. If we haven't reached size 17, try backtracking for size 17
    if best_size < 17 and time.time() < deadline - 5:
        time_limit = min(25, deadline - 5 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17
    
    # 4. Final local search pass
    if best_size < 17 and time.time() < deadline - 3:
        best_S = _local_search(best_S, universe, deadline)
        best_size = len(best_S)
    
    return best_S
