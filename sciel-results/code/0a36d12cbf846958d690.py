import numpy as np
import random

def construct():
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = 0
    best_graph = []
    
    # Number of trials
    trials = 200
    
    for _ in range(trials):
        random.shuffle(all_pairs)
        
        # Adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        edges = set()
        
        for u, v in all_pairs:
            # Check if adding (u, v) creates a 4-cycle
            # We need to check if there exist w, x such that:
            # adj[u, w] == 1 and adj[v, x] == 1 and adj[w, x] == 1
            # This forms cycle u-v-x-w-u
            
            # Get neighbors of u and v
            # Using np.where is fast enough for n=20
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            # If either has no neighbors, no 4-cycle can be formed
            if len(nu) == 0 or len(nv) == 0:
                safe = True
            else:
                # Check submatrix adj[nu, nv] for any 1
                # Note: nu and nv are indices
                if np.any(adj[np.ix_(nu, nv)]):
                    safe = False
                else:
                    safe = True
            
            if safe:
                # Add edge
                edges.add((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = list(edges)
            
    result = []
    for u, v in best_graph:
        result.append([u, v, 0])
        
    return result
