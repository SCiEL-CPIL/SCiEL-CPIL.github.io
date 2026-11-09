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

def _try_add_to_sums(sums, subset):
    """
    Try to add all elements of subset to the existing sums set.
    Returns (new_sums, success). If success is False, a collision was found
    and new_sums is the original sums (unchanged).
    """
    current = set(sums)
    for x in subset:
        v = _LCM // x
        new_entries = []
        for s in current:
            ns = s + v
            if ns in current:
                return sums, False
            new_entries.append(ns)
        current |= set(new_entries)
    return current, True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    # 1. Greedy construction: Add largest numbers first
    # This tends to produce a large dissociated set because large numbers
    # have smaller weights (L/i) and are less likely to create small-integer collisions.
    S = []
    sums = {0}
    
    # Sort in descending order
    candidates = sorted(universe, reverse=True)
    
    for x in candidates:
        if time.time() > deadline:
            break
        v = _LCM // x
        # Check if adding x creates a collision
        # New sums would be {s + v for s in sums}
        # Collision if sums & {s + v for s in sums} is not empty
        # Or simply: if (s + v) in sums for any s in sums
        # Note: s=0 is in sums. So if v in sums, collision.
        # We need to check all existing sums.
        
        # Optimization: Just generate the new set and check intersection
        # But we can break early if we find a collision.
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums |= {s + v for s in sums}

    best_S = list(S)
    best_size = len(best_S)
    
    # 2. Local Search: Try to increase size.
    # Phase A: 1-for-2 swaps (remove 1, add 2) -> size +1
    # Phase B: 2-for-3 swaps (remove 2, add 3) -> size +1 (more flexible)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        found_better = False
        
        # Phase A: 1-for-2 moves
        moves_a = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves_a.append((a, x, y))
        random.shuffle(moves_a)
        
        for a, x, y in moves_a:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                found_better = True
                break
        
        if found_better:
            continue
        
        # Phase B: 2-for-3 moves (only if we haven't found a 1-for-2)
        if time.time() > deadline - 5:
            break
            
        moves_b = []
        for a, b in itertools.combinations(best_S, 2):
            for x, y, z in itertools.combinations(not_in_set, 3):
                moves_b.append((a, b, x, y, z))
        random.shuffle(moves_b)
        
        for a, b, x, y, z in moves_b:
            if time.time() > deadline - 5:
                break
            
            candidate = [c for c in best_S if c != a and c != b] + [x, y, z]
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                found_better = True
                break
        
        if found_better:
            continue
    
    return best_S
