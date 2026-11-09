import numpy as np
from itertools import combinations

def construct():
    # Paley graph on 41 vertices (41 is a prime, 41 ≡ 1 mod 4)
    # This gives a (5,5)-free graph with 41 vertices.
    # We then attempt to add a 42nd vertex to reach the known best lower bound R(5,5) >= 43.
    
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    
    adj = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Precompute all K4s and independent K4s (complements of K4s) in the base graph
    # A new vertex v creates a K5 iff it is adjacent to all vertices of some K4 in the base graph.
    # A new vertex v creates an independent 5-set iff it is non-adjacent to all vertices of some independent K4.
    
    # Find all 4-cliques
    k4s = []
    for combo in combinations(range(q), 4):
        is_clique = True
        for a in range(4):
            for b in range(a + 1, 4):
                if not adj[combo[a], combo[b]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            k4s.append(combo)
    
    # Find all independent 4-sets
    indep4s = []
    for combo in combinations(range(q), 4):
        is_indep = True
        for a in range(4):
            for b in range(a + 1, 4):
                if adj[combo[a], combo[b]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            indep4s.append(combo)
    
    # Try to add a 42nd vertex
    # We need to find an adjacency vector x in {0,1}^41 such that:
    #   For all K4s C: not (x[c] == 1 for all c in C)
    #   For all indep4s C: not (x[c] == 0 for all c in C)
    
    # Use random search with constraint checking
    # To speed up, we can use a simple local search or just random with early termination.
    # Given the structure of Paley graphs, a random vector might work, or we can try
    # vectors that are "balanced" or derived from the graph structure.
    
    # First, try a few structured vectors:
    # 1. All zeros (vertex connected to no one) - creates independent 5-set if there's an indep4
    # 2. All ones - creates K5 if there's a K4
    # 3. Random vectors
    
    import random
    random.seed(12345)
    
    found = False
    best_x = None
    
    # Try random vectors
    for attempt in range(50000):
        x = np.random.randint(0, 2, size=q)
        
        valid = True
        # Check K4 constraints
        for c in k4s:
            if all(x[v] == 1 for v in c):
                valid = False
                break
        if not valid:
            continue
        
        # Check independent 4-set constraints
        for c in indep4s:
            if all(x[v] == 0 for v in c):
                valid = False
                break
        if not valid:
            continue
        
        found = True
        best_x = x
        break
    
    if found:
        n = q + 1
        # Build the full adjacency matrix
        adj_full = np.zeros((n, n), dtype=np.int8)
        adj_full[:q, :q] = adj
        adj_full[:q, q] = best_x
        adj_full[q, :q] = best_x
        adj_full[q, q] = 0
    else:
        n = q
        adj_full = adj
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_full[i, j]:
                edges.append((i, j))
    
    return n, edges
