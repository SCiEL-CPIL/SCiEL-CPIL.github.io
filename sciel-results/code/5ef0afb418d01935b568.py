import random
import math
from math import gcd

def construct():
    n = 128
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    bad_triples.append((a, b, c))
    
    # Map number to list of bad triple indices
    vertex_triples = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(bad_triples):
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)

    best_size = 0
    best_set = set()
    
    # Local Search / Simulated Annealing style
    # We will perform several restarts
    for restart in range(200): # 200 restarts should be fast enough
        # Initialize S as all numbers
        S = set(range(1, n + 1))
        R = set() # Removed
        
        # Step 1: Greedily remove to fix conflicts
        # We maintain a list of active bad triples
        # Instead of a full list, we can just check on the fly or maintain a count
        # Since N is small, we can just find a bad triple, remove one, repeat.
        
        # To make it efficient, we keep a set of "active" triple indices
        active_triples = set(range(len(bad_triples)))
        
        while active_triples:
            # Pick a random active triple
            t_idx = random.choice(list(active_triples))
            a, b, c = bad_triples[t_idx]
            
            # Remove one of a, b, c
            # Heuristic: Remove the one with the most active triples (highest degree)
            # Or just random? Random is often better for escaping local minima.
            # Let's try random first.
            v = random.choice([a, b, c])
            
            S.remove(v)
            R.add(v)
            
            # Update active_triples: remove all triples containing v
            # This is O(deg(v))
            for idx in vertex_triples[v]:
                if idx in active_triples:
                    active_triples.remove(idx)
        
        # Step 2: Try to add back removed elements
        # Sort R by some heuristic? Or random order?
        # Random order allows different configurations.
        r_list = list(R)
        random.shuffle(r_list)
        
        for v in r_list:
            # Check if v can be added
            # v creates a bad triple if there exist x, y in S such that (v,x,y) is bad.
            # We can precompute "conflict pairs" for v?
            # Or just check all pairs in S.
            # |S| is around 80-100. Pairs ~ 4000.
            # 128 * 4000 = 500k checks per restart.
            # 200 restarts = 100M checks. Might be too slow in Python.
            
            # Optimization: Precompute for each v, the set of pairs (x,y) that form a bad triple with v.
            # Let's do this once outside the loop.
            pass # Placeholder
            
        # If we implement the optimization, this will be fast.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    return list(best_set)
