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
        
    # Order universe to try larger numbers first (smaller scaled values)
    # This often leads to faster growth of the sum set without collisions?
    # Or maybe smaller numbers first?
    # Let's try sorting by scaled value descending (i.e., x ascending) or ascending (x descending).
    # Let's try a few orders.
    
    # We will try to build the set of size target_size.
    # We use a recursive helper.
    
    start_time = time.time()
    
    def dfs(current_set, current_sums, remaining, depth):
        nonlocal start_time
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_set) == target_size:
            return list(current_set)
            
        # Heuristic: Try candidates in a specific order
        # We can shuffle the remaining candidates if we want randomization,
        # but that's expensive. Instead, we can randomize the initial universe order 
        # outside and pass it down, or just rely on the order.
        
        # To increase diversity, we can try different orders of 'remaining'
        # But standard DFS with a fixed order is deterministic.
        # We'll call this function multiple times with different initial orders.
        
        for i, x in enumerate(remaining):
            v = _LCM // x
            new_sums = set()
            is_valid = True
            # Check collision
            # Optimization: if (s+v) in current_sums for any s, it's invalid.
            # We can do this faster?
            # current_sums is a set.
            for s in current_sums:
                if (s + v) in current_sums:
                    is_valid = False
                    break
            
            if is_valid:
                # Calculate new sums
                # new_sums = current_sums | {s + v for s in current_sums}
                # We need to update current_sums for the next level
                # But we don't want to modify it in place if we backtrack.
                # So we create a new set.
                # This is O(2^depth).
                
                # Optimization: Check validity first, then compute sums.
                # If valid, compute new_sums.
                new_sums = current_sums | {s + v for s in current_sums}
                
                # Recurse
                next_remaining = remaining[:i] + remaining[i+1:]
                result = dfs(current_set + [x], new_sums, next_remaining, depth + 1)
                if result is not None:
                    return result
                    
        return None

    # Try a few random orders of the universe
    # We want to find ANY set of size target_size.
    # Randomizing the order helps escape local minima in the search tree.
    
    for attempt in range(100): # Max 100 attempts
        if time.time() - start_time > time_limit:
            break
            
        order = universe.copy()
        random.shuffle(order)
        
        # We pass the full universe as 'remaining'
        # But we need to ensure we pick exactly target_size elements.
        # The DFS above picks from remaining until current_set reaches target_size.
        # This works.
        
        result = dfs([], {0}, order, 0)
        if result is not None:
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    best_S = []
    best_size = 0
    
    # 1. Try multiple greedy orderings to find a strong initial set
    # Deterministic orders with different strategies
    orders = [
        sorted(universe, reverse=True),  # Largest first
        sorted(universe),                 # Smallest first
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
    ]
    
    # Random orders
    for _ in range(40):
        random_order = universe.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 20:
            break
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # 2. Local Search: Try multiple improvement strategies
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # Strategy A: Try to add a single element
        if not_in_set:
            for x in not_in_set:
                if time.time() > deadline - 5:
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
                if time.time() > deadline - 5:
                    break
                
                candidate = best_S[:idx] + best_S[idx+1:] + [x, y]
                
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
        
        # Strategy C: Try 1-for-2 swaps (remove 2, add 1) to improve structure
        if not improved and len(best_S) >= 2:
            moves = []
            for i, j in itertools.combinations(range(len(best_S)), 2):
                for x in not_in_set:
                    moves.append((i, j, x))
            
            random.shuffle(moves)
            
            for i, j, x in moves:
                if time.time() > deadline - 5:
                    break
                
                # Remove elements at positions i and j, add x
                new_S = []
                for k, elem in enumerate(best_S):
                    if k != i and k != j:
                        new_S.append(elem)
                new_S.append(x)
                
                if _is_dissociated(new_S):
                    # This doesn't increase size, but might enable future additions
                    # Only accept if it allows adding more elements later
                    # For now, skip this as it doesn't directly help maximize size
                    pass

    # 3. If we haven't reached size 17, try backtracking for size 17
    # We know max size is likely 16 or 17.
    # If best_size < 17, we try to find a set of size 17.
    if best_size < 17 and time.time() < deadline - 5:
        # Give it 30 seconds to find a size 17 set
        time_limit = min(30, deadline - 5 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17

    return best_S
