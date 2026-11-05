import time
import random

_DEADLINE = 50.0
_N = 256

def _is_good_pair(a, b):
    return (a * b) % (a + b) != 0

def _build_conflict_graph():
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict, order):
    S = set()
    for x in order:
        if not (S & conflict[x]):
            S.add(x)
    return S

def _randomized_greedy(conflict, time_limit):
    start = time.time()
    best = set()
    best_size = 0
    
    # Run multiple randomized greedy passes
    while time.time() - start < time_limit:
        order = list(range(1, _N + 1))
        random.shuffle(order)
        S = _greedy_build(conflict, order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
            
        # Early exit if we hit a theoretical upper bound or perfect score (unlikely)
        # Just let it run until time limit for now
        
    return best

def construct():
    start = time.time()
    conflict = _build_conflict_graph()
    
    # Allocate 90% of the deadline for search
    search_time = _DEADLINE * 0.9
    
    # Start with a deterministic good heuristic (odds) to have a baseline
    S_odd = set(range(1, _N + 1, 2))
    # Try to add evens greedily to the odd set? No, just use randomized greedy from scratch
    # Randomized greedy is often better than heuristic + local search for MIPs
    
    best_set = _randomized_greedy(conflict, search_time)
    
    # If no set found (should not happen), fallback
    if not best_set:
        best_set = {1}
        
    result = sorted(best_set)
    if not result:
        return [1]
    return result
