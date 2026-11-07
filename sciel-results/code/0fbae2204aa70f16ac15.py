import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for the remaining graph
    # We need to track which edges are still "active"
    # Using a set of frozensets for edges for O(1) lookup/removal
    active_edges = set(frozenset(e) for e in edges)
    
    # Adjacency list for active graph
    adj = [set() for _ in range(n)]
    for e in active_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    bicliques = []
    
    while active_edges:
        if time.time() > deadline:
            break
            
        # Find vertex with maximum degree in active graph
        max_deg = -1
        u = -1
        for i in range(n):
            deg = len(adj[i])
            if deg > max_deg:
                max_deg = deg
                u = i
                if deg == 0:
                    break # Should not happen if active_edges is not empty, but safe
            
        if u == -1:
            break
            
        # Initialize biclique with L={u}, R=adj[u]
        L = [u]
        R = set(adj[u])
        
        # Try to expand L by adding vertices v such that R is a subset of adj[v]
        # i.e., v is connected to all nodes in R
        for v in range(n):
            if v == u:
                continue
            if len(adj[v]) == 0:
                continue
            # Check if R is subset of adj[v]
            # This is O(|R|)
            # Optimization: if len(adj[v]) < len(R), skip
            if len(adj[v]) >= len(R):
                # Check subset
                # Since R is a set, we can do R.issubset(adj[v])
                if R.issubset(adj[v]):
                    L.append(v)
                    
        # The biclique is (L, R)
        # Edges covered are all (l, r) for l in L, r in R
        # We need to remove these from active_edges
        # And update adj lists
        
        # Generate all edges in the biclique
        biclique_edges = []
        for l in L:
            for r in R:
                if l != r: # Bicliques usually don't have self-loops, and graph is simple
                    biclique_edges.append(frozenset((l, r)))
        
        # Remove these edges from active graph
        for e in biclique_edges:
            if e in active_edges:
                active_edges.remove(e)
                u, v = tuple(e)
                adj[u].remove(v)
                adj[v].remove(u)
                
        # Store the biclique
        # Convert sets/lists to plain lists of ints for return format
        bicliques.append([list(L), list(R)])
        
    # Fallback: if we ran out of time or failed to cover all edges,
    # we must return a valid partition. 
    # If active_edges is not empty, we can add single-edge bicliques for the rest.
    if active_edges:
        for e in active_edges:
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            
    # Final check and cleanup
    # Ensure types are correct
    final_result = []
    for b in bicliques:
        L = [int(x) for x in b[0]]
        R = [int(x) for x in b[1]]
        final_result.append([L, R])
        
    return final_result
