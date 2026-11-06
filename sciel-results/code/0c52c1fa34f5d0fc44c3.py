"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
from math import lcm

def construct():
    n = 128
    # Precompute LCM table for speed
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm(i, j)
    
    # Precompute bad triples
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            ab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    
    # For each element, count bad triples
    triple_count = [0] * (n+1)
    for (a, b, c) in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Greedy: process in order of increasing triple_count
    order = sorted(range(1, n+1), key=lambda x: (triple_count[x], x))
    
    selected = []
    selected_set = set()
    
    for x in order:
        conflict = False
        for (a, b) in combinations(selected, 2):
            if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                conflict = True
                break
        if not conflict:
            selected.append(x)
            selected_set.add(x)
    
    # Local search: try to add elements that were skipped
    improved = True
    while improved:
        improved = False
        for x in range(1, n+1):
            if x in selected_set:
                continue
            conflict = False
            for (a, b) in combinations(selected, 2):
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    conflict = True
                    break
            if not conflict:
                selected.append(x)
                selected_set.add(x)
                improved = True
    
    if not selected:
        selected = [1]
    
    return selected
# EVOLVE-BLOCK-END
