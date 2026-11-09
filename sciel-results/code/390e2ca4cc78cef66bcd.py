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
            for c in range(b + 1, n + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    bad_triples.append((a, b, c))
    return bad_triples

def _greedy_with_conflicts(bad_triples, max_n, deadline):
    """
    Greedy hitting set approach: start with all elements, remove elements
    to break bad triples. At each step, pick the element that appears in
    the most remaining bad triples.
    """
    # Build conflict graph: for each element, which bad triples it's in
    from collections import defaultdict
    
    element_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        element_to_triples[a].append(idx)
        element_to_triples[b].append(idx)
        element_to_triples[c].append(idx)
    
    # Track which triples are still "active" (not yet hit)
    active = [True] * len(bad_triples)
    removed = set()
    selected = set(range(1, max_n + 1))
    
    # Count active triples per element
    active_count = {elem: len(triples) for elem, triples in element_to_triples.items()}
    
    while True:
        if time.time() > deadline:
            break
        
        # Find element with max active triple count
        best_elem = None
        best_count = 0
        for elem in selected:
            if elem not in removed:
                cnt = 0
                for tidx in element_to_triples.get(elem, []):
                    if active[tidx]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_elem = elem
        
        if best_elem is None or best_count == 0:
            break
        
        # Remove this element
        removed.add(best_elem)
        selected.discard(best_elem)
        
        # Deactivate all triples containing this element
        for tidx in element_to_triples.get(best_elem, []):
            if active[tidx]:
                active[tidx] = False
    
    return list(selected)

def _local_search(initial_set, bad_triples, max_n, deadline):
    """
    Local search: try to add back removed elements or swap elements
    to increase set size while maintaining no bad triples.
    """
    from collections import defaultdict
    
    selected = set(initial_set)
    
    # Precompute for quick conflict checking
    # For each pair, check if adding a third element creates a bad triple
    # We'll use the bad_triples list to check conflicts
    
    # Build a set of bad triples for O(1) lookup
    bad_set = set(bad_triples)
    
    # Also build: for each pair (a,b), which c values form bad triples
    pair_to_thirds = defaultdict(set)
    for a, b, c in bad_triples:
        pair_to_thirds[(a, b)].add(c)
        pair_to_thirds[(b, a)].add(c)
        pair_to_thirds[(a, c)].add(b)
        pair_to_thirds[(c, a)].add(b)
        pair_to_thirds[(b, c)].add(a)
        pair_to_thirds[(c, b)].add(a)
    
    def can_add(elem):
        """Check if elem can be added to selected without creating a bad triple."""
        for other in selected:
            if (elem, other) in pair_to_thirds:
                for third in pair_to_thirds[(elem, other)]:
                    if third in selected:
                        return False
        return True
    
    # Try to add elements not in selected
    candidates = [x for x in range(1, max_n + 1) if x not in selected]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in candidates:
            if elem in selected:
                continue
            if can_add(elem):
                selected.add(elem)
                improved = True
    
    # Try swaps: remove one element, add two others
    # This is more complex; for now, just try single additions
    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    # Strategy 1: Greedy hitting set (remove elements to break all bad triples)
    result1 = _greedy_with_conflicts(bad_triples, max_n, deadline)
    
    # Strategy 2: Greedy forward selection (original approach but optimized)
    lcm_table = _precompute_lcms(max_n)
    selected2 = []
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        valid = True
        for i in range(len(selected2)):
            for j in range(i + 1, len(selected2)):
                a, b = selected2[i], selected2[j]
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected2.append(x)
    
    # Pick the better result
    if len(result1) >= len(selected2):
        best = result1
    else:
        best = selected2
    
    # Try local search to improve
    if time.time() < deadline - 5:
        best = _local_search(best, bad_triples, max_n, deadline)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
