"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, use 60 for safety margin against 90s kill
    
    edges = P['edges']
    n = P['n']
    
    # Initial adjacency list
    adj = [set() for _ in range(n)]
    remaining_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        remaining_edges.add(frozenset((u, v)))
    
    bicliques = []
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: return whatever we have, but it must be valid.
            # If we break early, we haven't partitioned all edges.
            # The task requires partitioning ALL edges.
            # So we must finish. The greedy is fast, so this shouldn't happen.
            # If it does, we might need a faster fallback.
            # For now, assume it finishes.
            pass
        
        # Pick a vertex with maximum degree
        u = max(range(n), key=lambda x: len(adj[x]))
        
        # Initialize biclique with u
        L = {u}
        # R starts as all neighbors of u
        R = set(adj[u])
        
        # Expand L and R
        changed = True
        while changed:
            changed = False
            
            # Try to add to L
            # A vertex w can be added to L if R is a subset of adj[w]
            # We only consider vertices not in L or R
            candidates_L = [w for w in range(n) if w not in L and w not in R]
            for w in candidates_L:
                if R.issubset(adj[w]):
                    L.add(w)
                    changed = True
            
            # Try to add to R
            # A vertex z can be added to R if L is a subset of adj[z]
            candidates_R = [z for z in range(n) if z not in L and z not in R]
            for z in candidates_R:
                if L.issubset(adj[z]):
                    R.add(z)
                    changed = True
        
        # Ensure L and R are not empty
        if not L or not R:
            # Fallback for isolated edge or issue
            # Pick any remaining edge
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            L = {u}
            R = {v}
        
        # Add biclique
        bicliques.append([list(L), list(R)])
        
        # Remove edges between L and R
        to_remove = set()
        for u in L:
            for v in R:
                if u != v:
                    to_remove.add(frozenset((u, v)))
        
        for e in to_remove:
            if e in remaining_edges:
                remaining_edges.remove(e)
                u, v = tuple(e)
                adj[u].remove(v)
                adj[v].remove(u)
    
    # Convert to required format: list of [left, right]
    # Types must be Python int and list/tuple
    result = []
    for bic in bicliques:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
