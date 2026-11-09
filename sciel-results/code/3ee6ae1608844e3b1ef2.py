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

def _local_search(S, universe, deadline):
    """
    Perform local search to maximize cardinality.
    Tries 1-for-2, 2-for-3, and 1-for-1 (escape) moves.
    """
    best_S = list(S)
    best_size = len(best_S)
    
    while time.time() < deadline:
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        improved = False
        
        # Phase 1: 1-for-2 swaps
        if len(not_in_set) >= 2:
            moves_a = [(a, x, y) for a in best_S for x, y in itertools.combinations(not_in_set, 2)]
            random.shuffle(moves_a)
            
            for a, x, y in moves_a:
                if time.time() > deadline:
                    return best_S
                
                candidate = [z for z in best_S if z != a] + [x, y]
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            continue
        
        # Phase 2: 2-for-3 swaps
        if len(not_in_set) >= 3:
            moves_b = [(a, b, x, y, z) for a, b in itertools.combinations(best_S, 2) 
                       for x, y, z in itertools.combinations(not_in_set, 3)]
            random.shuffle(moves_b)
            
            for a, b, x, y, z in moves_b:
                if time.time() > deadline:
                    return best_S
                
                candidate = [c for c in best_S if c != a and c != b] + [x, y, z]
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            continue
        
        # Phase 3: 1-for-1 swaps (Escape local optimum)
        # Try to find a swap that leads to a 1-for-2 opportunity
        escaped = False
        for _ in range(50):
            if time.time() > deadline:
                return best_S
            
            if not best_S or not not_in_set:
                break
                
            a = random.choice(best_S)
            x = random.choice(not_in_set)
            
            # Form candidate with swap
            candidate = [z for z in best_S if z != a] + [x]
            
            # Check if candidate is valid
            if not _is_dissociated(candidate):
                continue
                
            # Check if candidate allows a 1-for-2
            in_cand = set(candidate)
            not_in_cand = [z for z in universe if z not in in_cand]
            
            if len(not_in_cand) < 2:
                continue
                
            # Sample a few 1-for-2 moves from candidate
            moves_from_cand = [(c, u, v) for c in candidate for u, v in itertools.combinations(not_in_cand, 2)]
            random.shuffle(moves_from_cand)
            
            for c, u, v in moves_from_cand[:10]:
                if time.time() > deadline:
                    return best_S
                    
                final_candidate = [z for z in candidate if z != c] + [u, v]
                if _is_dissociated(final_candidate):
                    best_S = final_candidate
                    best_size += 1
                    improved = True
                    escaped = True
                    break
                    
            if escaped:
                break
        
        if not escaped:
            # Stuck
            break
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85
    universe = list(range(1, 23))
    
    best_overall = []
    best_size = 0
    
    # Multiple greedy starts
    strategies = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: random.random()),
        sorted(universe, key=lambda x: random.random())
    ]
    
    for strat in strategies:
        if time.time() > deadline - 15:
            break
            
        S = []
        sums = {0}
        
        for x in strat:
            if time.time() > deadline - 15:
                break
            v = _LCM // x
            is_valid = True
            for s in sums:
                if (s + v) in sums:
                    is_valid = False
                    break
            if is_valid:
                S.append(x)
                sums |= {s + v for s in sums}
                
        # Local Search
        S_ls = _local_search(S, universe, deadline - 5)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_overall = S_ls
            
    return best_overall
