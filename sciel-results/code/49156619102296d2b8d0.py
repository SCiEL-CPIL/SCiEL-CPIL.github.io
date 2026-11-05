"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 128
    # 1. Identify all "bad" triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We store for each element v, the set of pairs (a, b) such that {v, a, b} is a bad triple.
    # This allows fast checking during the greedy phase.
    
    # To save memory and time, we can just store the list of bad triples first,
    # then map them.
    bad_triples = []
    
    # Optimization: Pre-compute lcms for all pairs?
    # 128*128/2 = 8192 pairs. Very fast.
    lcms = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[(i, j)] = lcm(i, j)
    
    # Find bad triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[(a, b)]
            # c must be > b
            for c in range(b + 1, N + 1):
                # Check if lcm(a,c) == lab and lcm(b,c) == lab
                # We can compute lcm(a,c) and lcm(b,c) on the fly or precompute.
                # Precomputing all 8192 is better.
                if (a, c) not in lcms:
                    lcms[(a, c)] = lcm(a, c)
                if (b, c) not in lcms:
                    lcms[(b, c)] = lcm(b, c)
                    
                lac = lcms[(a, c)]
                lbc = lcms[(b, c)]
                
                if lab == lac == lbc:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return list(range(64, 129)) # Fallback

    # Map each element to the pairs it conflicts with
    # conflict_map[v] = set of (u, w) such that {v, u, w} is a bad triple
    conflict_map = defaultdict(set)
    for a, b, c in bad_triples:
        conflict_map[a].add((b, c))
        conflict_map[b].add((a, c))
        conflict_map[c].add((a, b))
        
    if time.time() > deadline:
        return list(range(64, 129))

    # Calculate degree of each node (number of bad triples it is in)
    degrees = {i: len(conflict_map[i]) for i in range(1, N + 1)}
    
    # Sort nodes by degree ascending (pick least conflicting first)
    sorted_nodes = sorted(range(1, N + 1), key=lambda x: degrees[x])
    
    selected = []
    selected_set = set()
    
    # Greedy selection
    for v in sorted_nodes:
        if time.time() > deadline:
            break
            
        # Check if v can be added
        # v can be added if no pair (u, w) in conflict_map[v] is already in selected_set
        valid = True
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                valid = False
                break
                
        if valid:
            selected.append(v)
            selected_set.add(v)
            
    # Ensure non-empty
    if not selected:
        selected.append(1)
        
    return selected

# EVOLVE-BLOCK-END
