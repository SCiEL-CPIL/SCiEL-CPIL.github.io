"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """Precompute all bad triples (a,b,c) with 1<=a<b<c<=max_n."""
    n = max_n
    # Precompute lcm table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_conflict_map(bad_triples, max_n):
    """For each element, find which other elements it appears in bad triples with."""
    # conflict[a] = set of elements that with some third element form a bad triple with a
    # Actually, we need: for a candidate x, and a pair (a,b) already selected,
    # check if (a,b,x) is a bad triple.
    # Better: for each pair (a,b), store the set of c such that (a,b,c) is bad.
    # Then when considering adding x, check if for any pair (a,b) in selected, x is in bad_pairs[(a,b)].
    
    # Alternative: for each element x, precompute the set of pairs (a,b) such that {a,b,x} is a bad triple.
    # Then x can be added iff no such pair (a,b) is fully in selected.
    
    # Let's build: for each x, list of (a,b) pairs that with x form a bad triple.
    # And also a global set of bad triple frozensets for O(1) lookup.
    
    bad_set = set()
    for (a, b, c) in bad_triples:
        bad_set.add(frozenset((a, b, c)))
    
    # For each element, the set of pairs that conflict with it
    elem_conflicts = {i: [] for i in range(1, max_n + 1)}
    for (a, b, c) in bad_triples:
        elem_conflicts[a].append((b, c))
        elem_conflicts[b].append((a, c))
        elem_conflicts[c].append((a, b))
    
    return bad_set, elem_conflicts

def _greedy_with_order(order, bad_set, elem_conflicts):
    """Greedy: process elements in given order, add if no bad triple completed."""
    selected = set()
    for x in order:
        # Check if adding x creates any bad triple with pairs in selected
        can_add = True
        for (a, b) in elem_conflicts[x]:
            if a in selected and b in selected:
                can_add = False
                break
        if can_add:
            selected.add(x)
    return selected

def _local_search(selected, bad_set, elem_conflicts, max_n):
    """Try to improve by adding elements or swapping."""
    selected = set(selected)
    improved = True
    while improved:
        improved = False
        # Try adding each unselected element
        for x in range(1, max_n + 1):
            if x in selected:
                continue
            can_add = True
            for (a, b) in elem_conflicts[x]:
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
                improved = True
        # Try swaps: remove one, add another
        # This is more expensive; do a limited version
        # Actually, let's try: for each x not in selected, see if removing one element y from selected
        # allows x to be added.
        for x in range(1, max_n + 1):
            if x in selected:
                continue
            # Find which pairs block x
            blocking_pairs = []
            for (a, b) in elem_conflicts[x]:
                if a in selected and b in selected:
                    blocking_pairs.append((a, b))
            if not blocking_pairs:
                continue  # would have been added already
            # Try removing one element from a blocking pair
            for (a, b) in blocking_pairs:
                for y in (a, b):
                    if y in selected:
                        new_selected = selected - {y}
                        new_selected.add(x)
                        # Verify no bad triples in new_selected
                        valid = True
                        for triple in bad_set:
                            if triple.issubset(new_selected):
                                valid = False
                                break
                        if valid and len(new_selected) > len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline for safety
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    bad_set, elem_conflicts = _build_conflict_map(bad_triples, max_n)
    
    best = set()
    
    # First, do a deterministic ascending greedy as baseline
    order = list(range(1, max_n + 1))
    best = _greedy_with_order(order, bad_set, elem_conflicts)
    best = _local_search(best, bad_set, elem_conflicts, max_n)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return sorted(best)
    
    # Randomized restarts
    rng = random.Random(42)
    for restart in range(5000):
        if time.time() > deadline - 3:
            break
        
        # Random order
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        
        candidate = _greedy_with_order(order, bad_set, elem_conflicts)
        candidate = _local_search(candidate, bad_set, elem_conflicts, max_n)
        
        if len(candidate) > len(best):
            best = candidate
    
    # Final local search on best
    if time.time() < deadline - 1:
        best = _local_search(best, bad_set, elem_conflicts, max_n)
    
    return sorted(best)
# EVOLVE-BLOCK-END
