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

def _local_search(conflict, S, time_limit):
    S = set(S)
    start = time.time()
    improved = True
    
    while improved and time.time() - start < time_limit:
        improved = False
        # Identify candidates for removal: u in S
        # For each u, find candidates for addition: v not in S such that N_S(v) subset of {u}
        # This means v is only connected to u in S.
        candidates_by_remover = {u: [] for u in S}
        
        for v in range(1, _N + 1):
            if v in S:
                continue
            neighbors_in_S = conflict[v] & S
            if len(neighbors_in_S) == 1:
                u = next(iter(neighbors_in_S))
                candidates_by_remover[u].append(v)
        
        # Try to find a pair (v1, v2) for some u
        # We want to remove u and add v1, v2.
        # v1 and v2 must be in candidates_by_remover[u].
        # And v1, v2 must not be adjacent to each other.
        
        for u in S:
            cands = candidates_by_remover[u]
            if len(cands) < 2:
                continue
            
            for i in range(len(cands)):
                v1 = cands[i]
                for j in range(i + 1, len(cands)):
                    v2 = cands[j]
                    if v1 not in conflict[v2]:
                        S.remove(u)
                        S.add(v1)
                        S.add(v2)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
    return S

def _randomized_greedy(conflict, time_limit):
    start = time.time()
    best = set()
    best_size = 0
    
    # Run multiple randomized greedy passes
    while time.time() - start < time_limit:
        # Mix random and min-degree orderings
        mode = random.choice([0, 1, 1]) # Weight min-degree higher
        
        if mode == 0:
            order = list(range(1, _N + 1))
            random.shuffle(order)
        else:
            # Min degree first
            order = sorted(range(1, _N + 1), key=lambda x: len(conflict[x]))
            # Add some randomness to tie-breaking or slight perturbation?
            # Just shuffling the sorted list might break the heuristic.
            # Let's stick to pure min-degree for mode 1, but maybe randomize if degrees are equal?
            # For now, pure sorted is fine.
        
        S = _greedy_build(conflict, order)
        
        # Local Search
        # Allocate a small chunk of time for local search per restart
        # If we have 50s, and ~100 restarts, 0.2s per local search is 20s total.
        # This leaves 30s for greedy and overhead.
        S = _local_search(conflict, S, 0.2)
        
        if len(S) > best_size:
            best = S
            best_size = len(S)
            
    return best

def construct():
    start = time.time()
    conflict = _build_conflict_graph()
    
    # Allocate 90% of the deadline for search
    search_time = _DEADLINE * 0.9
    
    best_set = _randomized_greedy(conflict, search_time)
    
    # If no set found (should not happen), fallback
    if not best_set:
        best_set = {1}
        
    result = sorted(best_set)
    if not result:
        return [1]
    return result
