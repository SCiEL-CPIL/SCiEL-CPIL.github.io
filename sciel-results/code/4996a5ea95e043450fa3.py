"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(max_n):
    """Precompute LCM table for all pairs in [1, max_n]."""
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    Returns a list of tuples (a,b,c).
    """
    n = max_n
    lcm_table = _precompute_lcms(n)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _greedy_with_conflicts(bad_triples, order):
    """
    Greedy: process elements in given order, add if no conflict with already selected.
    Conflict means there exists a bad triple containing this element and two already-selected elements.
    """
    # For each element, precompute the set of pairs (x,y) such that (x,y,elem) is a bad triple
    # Actually, we need: for element x, which pairs (a,b) in already_selected would form a bad triple with x?
    # Better: for each element x, store a set of frozensets {a,b} such that {a,b,x} is a bad triple.
    # Then when considering x, check if any such {a,b} is a subset of already_selected.
    
    # Precompute conflict map: for each x, list of (a,b) pairs where {a,b,x} is bad
    conflict_map = {i: [] for i in range(1, 129)}
    for (a, b, c) in bad_triples:
        conflict_map[a].append((b, c))
        conflict_map[b].append((a, c))
        conflict_map[c].append((a, b))
    
    selected = set()
    for x in order:
        ok = True
        for (a, b) in conflict_map[x]:
            if a in selected and b in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _local_search(selected, bad_triples, conflict_map, max_iters=1000):
    """
    Try to improve the set by:
    1. Adding elements that don't conflict
    2. Swapping: remove one element, add another that was previously conflicting
    """
    selected = set(selected)
    
    # Build conflict map for quick lookup
    # For each element not in selected, find which pairs in selected would conflict
    # Actually, let's do a simpler local search:
    
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        
        # Try to add any element not in selected
        for x in range(1, 129):
            if x in selected:
                continue
            ok = True
            for (a, b) in conflict_map[x]:
                if a in selected and b in selected:
                    ok = False
                    break
            if ok:
                selected.add(x)
                changed = True
        
        # Try swaps: remove one element, see if we can add more
        # For each element in selected, try removing it and then greedily adding
        # This is expensive, so let's do a simpler version:
        # For each element x not in selected, for each element y in selected,
        # check if removing y allows adding x (and possibly more)
        # This is O(n^2 * conflicts) which might be too slow, so skip for now
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Precompute
    bad_triples = _precompute_bad_triples(128)
    
    # Build conflict map
    conflict_map = {i: [] for i in range(1, 129)}
    for (a, b, c) in bad_triples:
        conflict_map[a].append((b, c))
        conflict_map[b].append((a, c))
        conflict_map[c].append((a, b))
    
    best_selected = set()
    best_size = 0
    
    # First, do a greedy pass with natural order
    order = list(range(1, 129))
    selected = _greedy_with_conflicts(bad_triples, order)
    best_selected = set(selected)
    best_size = len(selected)
    
    # Now do randomized restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Generate random order
        order = list(range(1, 129))
        random.shuffle(order)
        
        selected = _greedy_with_conflicts(bad_triples, order)
        
        # Local search: try to add any missing elements
        changed = True
        while changed:
            changed = False
            for x in range(1, 129):
                if x in selected:
                    continue
                ok = True
                for (a, b) in conflict_map[x]:
                    if a in selected and b in selected:
                        ok = False
                        break
                if ok:
                    selected.add(x)
                    changed = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # Check time
        if time.time() >= deadline - 5:
            break
    
    return list(best_selected)
# EVOLVE-BLOCK-END
