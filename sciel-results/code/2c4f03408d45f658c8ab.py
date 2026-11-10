"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute all bad triples: (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Use a set of frozensets for fast lookup
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.add((a, b, c))
    
    # Count how many bad triples each element participates in
    elem_bad_count = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        elem_bad_count[a] += 1
        elem_bad_count[b] += 1
        elem_bad_count[c] += 1
    
    # Start with all elements, iteratively remove elements from bad triples
    # Use a priority: remove element with highest bad-triple count
    in_set = [True] * (n + 1)
    remaining_bad = list(bad_triples)
    
    # Greedy removal: while bad triples exist, remove the element in the most bad triples
    while remaining_bad:
        if time.time() > deadline:
            break
        # Find the element that appears in the most remaining bad triples
        # To be efficient, count from remaining_bad
        counts = {}
        for (a, b, c) in remaining_bad:
            for x in (a, b, c):
                if in_set[x]:
                    counts[x] = counts.get(x, 0) + 1
        if not counts:
            break
        # Pick the element with max count
        max_elem = max(counts, key=lambda x: counts[x])
        in_set[max_elem] = False
        # Remove all bad triples containing max_elem
        remaining_bad = [(a, b, c) for (a, b, c) in remaining_bad if max_elem not in (a, b, c)]
    
    # Current solution
    current = [i for i in range(1, n + 1) if in_set[i]]
    best = set(current)
    
    # Try to add back removed elements
    removed = [i for i in range(1, n + 1) if not in_set[i]]
    
    # For each removed element, check if adding it back creates any bad triple
    # A bad triple with x exists if there are a,b in current such that (a,b,x) is bad
    # Precompute for each x, the set of pairs (a,b) that form bad triples with x
    bad_pairs_for = {i: set() for i in range(1, n + 1)}
    for (a, b, c) in bad_triples:
        bad_pairs_for[a].add((b, c))
        bad_pairs_for[b].add((a, c))
        bad_pairs_for[c].add((a, b))
    
    # Try to add back elements greedily
    for x in removed:
        if time.time() > deadline:
            break
        # Check if adding x to current creates any bad triple
        # Need: no (a,b) in current such that (a,b,x) is a bad triple
        # i.e., no (a,b) in bad_pairs_for[x] where both a,b are in current
        can_add = True
        for (a, b) in bad_pairs_for[x]:
            if in_set[a] and in_set[b]:
                can_add = False
                break
        if can_add:
            in_set[x] = True
            current.append(x)
            best.add(x)
    
    # Local search: try swapping one element in with one out
    # Try multiple rounds
    for _round in range(5):
        if time.time() > deadline:
            break
        improved = False
        for x in removed:
            if time.time() > deadline:
                break
            if in_set[x]:
                continue
            # Try adding x
            # Find which elements in current would conflict
            # We need to remove some elements to make room for x
            # Simple approach: try removing one element y from current such that
            # after removing y and adding x, no bad triples exist
            # This is complex; instead try: add x, then remove elements to fix
            # Simpler: just try to add x if it doesn't create conflicts (already done above)
            # For now, skip complex swaps
        # Try removing an element and seeing if we can add two others
        # This is getting complex; let's stick with the greedy approach
    
    result = [i for i in range(1, n + 1) if in_set[i]]
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
