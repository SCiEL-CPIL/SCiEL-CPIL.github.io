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

def _can_add(sums, v):
    """Check if value v can be added to the current sum set without collision."""
    shifted = {s + v for s in sums}
    return not (sums & shifted)

def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        if _can_add(sums, v):
            S.append(x)
            sums |= {s + v for s in sums}
    return S, sums

def _fill_set(S, sums, universe, time_limit):
    """Try to add any remaining elements that fit."""
    in_set = set(S)
    remaining = [x for x in universe if x not in in_set]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for x in remaining:
            if time.time() > time_limit:
                break
            v = _LCM // x
            if _can_add(sums, v):
                S.append(x)
                sums |= {s + v for s in sums}
                in_set.add(x)
                remaining.remove(x)
                improved = True
                break
    return S, sums

def _local_search(S, universe, time_limit):
    """Local search: try 0-for-1, 1-for-1, 1-for-2 moves."""
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        in_set = set(S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # Try 0-for-1: add a single element
        for x in not_in_set:
            if time.time() > time_limit:
                break
            candidate = S + [x]
            if _is_dissociated(candidate):
                S = candidate
                improved = True
                break
        
        if improved:
            continue
            
        # Try 1-for-1: swap one element for another
        for a in S:
            if time.time() > time_limit:
                break
            for x in not_in_set:
                if time.time() > time_limit:
                    break
                candidate = [z for z in S if z != a] + [x]
                if _is_dissociated(candidate):
                    S = candidate
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try 1-for-2: remove one, add two
        for a in S:
            if time.time() > time_limit:
                break
            for x, y in itertools.combinations(not_in_set, 2):
                if time.time() > time_limit:
                    break
                candidate = [z for z in S if z != a] + [x, y]
                if _is_dissociated(candidate):
                    S = candidate
                    improved = True
                    break
            if improved:
                break
        
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    best_S = []
    best_size = 0
    
    # 1. Try multiple greedy orderings to find a strong initial set
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: _LCM // x, reverse=True),
        sorted(universe, key=lambda x: _LCM // x),
    ]
    
    for _ in range(15):
        random_order = universe.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 20:
            break
        S, sums = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # 2. Fill phase: try to add any remaining elements
    if best_S and time.time() < deadline - 15:
        sums = _get_sums(best_S)
        best_S, _ = _fill_set(best_S, sums, universe, deadline - 15)
        best_size = len(best_S)
    
    # 3. Local Search
    if best_S and time.time() < deadline - 5:
        best_S = _local_search(best_S, universe, deadline - 5)
        best_size = len(best_S)
    
    if not best_S:
        best_S = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    return best_S
