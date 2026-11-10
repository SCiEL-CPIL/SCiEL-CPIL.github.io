"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import Counter, defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n+1))
    
    # Build mapping: element -> set of bad triples it's in
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    # Greedy hitting set: repeatedly remove the element that appears in the most uncovered triples
    removed = set()
    uncovered = set(bad_triples)
    
    # Precompute counts for efficiency
    counts = Counter()
    for x in range(1, n+1):
        counts[x] = len(elem_triples[x])
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        
        # Find element with max uncovered triple count
        best_x = None
        best_size = 0
        for x in range(1, n+1):
            if x in removed:
                continue
            size = len(elem_triples[x] & uncovered)
            if size > best_size:
                best_size = size
                best_x = x
        
        if best_x is None or best_size == 0:
            break
        
        removed.add(best_x)
        uncovered -= elem_triples[best_x]
    
    result = [x for x in range(1, n+1) if x not in removed]
    
    # Local search / improvement phase
    # Try to add back removed elements if possible
    if time.time() < deadline - 3:
        # For each removed element (in order of increasing triple involvement),
        # check if adding it back keeps the set valid
        removed_list = sorted(removed, key=lambda x: len(elem_triples[x]))
        for x in removed_list:
            if time.time() > deadline - 3:
                break
            # Check if adding x to result creates any bad triple
            # A bad triple with x would be (a, b, x) where a,b are in result
            # and lcm(a,b)==lcm(a,x)==lcm(b,x)
            can_add = True
            for a, b in [(a, b) for a in result for b in result if a < b]:
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    can_add = False
                    break
            if can_add:
                removed.discard(x)
                result.append(x)
                result.sort()
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
