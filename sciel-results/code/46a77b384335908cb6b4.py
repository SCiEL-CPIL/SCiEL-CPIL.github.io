import time
import itertools
import random

# LCM of 1..22
_LCM = 232792560

def _is_dissociated(subset):
    """Check if the set is dissociated using exact LCM arithmetic."""
    if len(subset) == 0:
        return True
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        # Check for collision
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums = sums | new_sums
    return len(sums) == (1 << len(subset))

def _greedy_construct(order, deadline):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        if time.time() > deadline - 1:
            break
        v = _LCM // x
        is_valid = True
        # Optimization: check intersection
        # We can just generate new_sums and check if any is in sums
        # If we do it this way, we only do it once per candidate
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums:
                is_valid = False
                break
            new_sums.add(ns)
        
        if is_valid:
            S.append(x)
            sums = sums | new_sums
    return S, sums

def _backtracking_search_size_17(universe, deadline):
    """
    Attempt to find a dissociated set of size 17 using backtracking.
    """
    start_time = time.time()
    
    # Heuristic: Sort by index descending? Or by weight?
    # Let's try multiple orderings
    orders = [
        sorted(universe, reverse=True), # 22..1
        sorted(universe),               # 1..22
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    ]
    
    # Add random orders
    for _ in range(10):
        o = universe.copy()
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        def dfs(current_set, current_sums, remaining_idx):
            if time.time() > deadline - 2:
                return None
                
            if len(current_set) == 17:
                return list(current_set)
                
            if len(current_set) + len(remaining_idx) < 17:
                return None
                
            # Try to pick next element
            # We can try all remaining elements
            # To improve pruning, we might want to try elements that are "safe" first
            # But we don't know which are safe without checking.
            
            # Optimization: If we are close to 17, we can be more selective?
            
            for i in range(len(remaining_idx)):
                x = remaining_idx[i]
                v = _LCM // x
                
                # Check if v can be added
                is_valid = True
                # We need to check if {s+v for s in current_sums} intersects current_sums
                # We can do this by iterating
                # If current_sums is large, this is slow.
                # But current_sums size is 2^k.
                
                # Quick check: if v is 0 (not possible) or if v is in sums? No.
                
                # Efficient check:
                # Create a set of new sums? No, just check intersection.
                # for s in current_sums: if s+v in current_sums: return False
                
                # This is O(2^k).
                
                # Let's just do it.
                # To speed up, we can break early.
                
                valid = True
                for s in current_sums:
                    if (s + v) in current_sums:
                        valid = False
                        break
                        
                if valid:
                    new_sums = current_sums | {s + v for s in current_sums}
                    next_remaining = remaining_idx[:i] + remaining_idx[i+1:]
                    result = dfs(current_set + [x], new_sums, next_remaining)
                    if result is not None:
                        return result
                        
            return None

        result = dfs([], {0}, order)
        if result is not None:
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    universe = list(range(1, 23))
    
    best_S = []
    best_size = 0
    
    # 1. Try to find size 17
    # This is the most valuable step.
    # We allocate 60 seconds for this.
    if time.time() < deadline - 25:
        result = _backtracking_search_size_17(universe, deadline - 25)
        if result is not None:
            best_S = result
            best_size = 17
    
    # 2. If size 17 not found, find best size 16
    if best_size < 17:
        # Try greedy with different orders
        orders = [
            sorted(universe, reverse=True),
            sorted(universe),
            [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        ]
        for _ in range(20):
            o = universe.copy()
            random.shuffle(o)
            orders.append(o)
            
        for order in orders:
            if time.time() > deadline - 5:
                break
            S, _ = _greedy_construct(order, deadline)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
                
        # Local search to try to reach 17 from 16
        if best_size == 16 and time.time() < deadline - 5:
            # Try 1-for-2 swaps
            in_set = set(best_S)
            not_in_set = [x for x in universe if x not in in_set]
            
            moves = [(a, x, y) for a in best_S for x, y in itertools.combinations(not_in_set, 2)]
            random.shuffle(moves)
            
            for a, x, y in moves:
                if time.time() > deadline - 2:
                    break
                candidate = [z for z in best_S if z != a] + [x, y]
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size = 17
                    break
                    
    # 3. Fallback
    if not best_S:
        best_S = [1, 2, 4, 8, 16]
        
    return best_S
