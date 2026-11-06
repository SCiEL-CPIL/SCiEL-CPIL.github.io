"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from itertools import combinations

def _find_bad_triples(n_max=128):
    """Find all triples (a,b,c) with a<b<c in [1,n_max] where lcm(a,b)=lcm(a,c)=lcm(b,c)."""
    bad = []
    lcm_cache = {}
    def get_lcm(a, b):
        key = (min(a,b), max(a,b))
        if key not in lcm_cache:
            lcm_cache[key] = lcm(a, b)
        return lcm_cache[key]
    
    for a in range(1, n_max+1):
        for b in range(a+1, n_max+1):
            ab = get_lcm(a, b)
            for c in range(b+1, n_max+1):
                if get_lcm(a, c) == ab and get_lcm(b, c) == ab:
                    bad.append((a, b, c))
    return bad

def _greedy_with_removal(bad_triples, n_max=128):
    """Start with all, remove vertices to eliminate all bad triples, minimizing removals."""
    # This is a hitting set / vertex cover problem on hypergraph
    # Use greedy: repeatedly pick the vertex in the most bad triples, remove it
    active_triples = list(bad_triples)
    removed = set()
    
    while active_triples:
        # Count occurrences
        counts = {}
        for (a,b,c) in active_triples:
            for v in (a,b,c):
                counts[v] = counts.get(v, 0) + 1
        # Pick vertex with max count
        v = max(counts, key=lambda x: counts[x])
        removed.add(v)
        # Remove all triples containing v
        active_triples = [t for t in active_triples if v not in t]
    
    # Result is all vertices not removed
    result = [i for i in range(1, n_max+1) if i not in removed]
    return result

def _local_improve(result, bad_triples, n_max=128, time_limit=60.0):
    """Try to add back removed vertices or swap to increase size."""
    start = time.time()
    current = set(result)
    
    def is_valid(s):
        for (a,b,c) in bad_triples:
            if a in s and b in s and c in s:
                return False
        return True
    
    # Try to add each missing vertex
    for v in range(1, n_max+1):
        if v in current:
            continue
        if time.time() - start > time_limit:
            break
        # Check if adding v creates any bad triple
        can_add = True
        for (a,b,c) in bad_triples:
            if v in (a,b,c):
                others = [x for x in (a,b,c) if x != v]
                if all(o in current for o in others):
                    can_add = False
                    break
        if can_add:
            current.add(v)
    
    return sorted(current)

def construct():
    deadline = time.time() + 85.0
    
    bad_triples = _find_bad_triples(128)
    
    if time.time() > deadline - 10.0:
        return list(range(1, 129))
    
    # Greedy vertex cover
    result = _greedy_with_removal(bad_triples, 128)
    
    # Local improvement
    result = _local_improve(result, bad_triples, 128, time_limit=70.0)
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
