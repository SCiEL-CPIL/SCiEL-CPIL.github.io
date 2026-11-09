import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 125

    def solve(n, seed):
        rng = random.Random(seed)
        
        # adj[i][j] = True if edge exists
        adj = np.zeros((n, n), dtype=bool)
        # common[i][j] = number of common neighbors of i and j
        common = np.zeros((n, n), dtype=np.int32)
        
        def add_edge(u, v):
            """Add edge (u,v) and update common counts. Returns False if it would create C4."""
            if u == v or adj[u][v]:
                return False
            
            # 1. Check if (u,v) creates a C4 with existing edges
            if np.sum(adj[u] & adj[v]) > 1:
                return False
            
            # 2. Check if adding (u,v) makes any other pair have > 1 common neighbor
            if np.any(common[u][adj[v]] > 0):
                return False
            
            # Similarly for v
            if np.any(common[v][adj[u]] > 0):
                return False
            
            # Safe to add
            adj[u][v] = adj[v][u] = True
            
            # Update common counts
            common[u, :] += adj[v, :]
            common[v, :] += adj[u, :]
            common[:, u] += adj[:, v]
            common[:, v] += adj[:, u]
            
            return True

        # Strategy: Greedy min-degree first
        # Repeatedly find vertex with min degree and try to add edges to it.
        
        while True:
            if time.time() > deadline:
                break
            
            degs = np.sum(adj, axis=1)
            min_deg = np.min(degs)
            min_vertices = np.where(degs == min_deg)[0]
            
            if len(min_vertices) == 0:
                break
                
            rng.shuffle(min_vertices)
            
            progress = False
            for u in min_vertices:
                non_neighbors = np.where(~adj[u])[0]
                non_neighbors = non_neighbors[non_neighbors != u]
                
                if len(non_neighbors) == 0:
                    continue
                
                rng.shuffle(non_neighbors)
                
                for v in non_neighbors:
                    if add_edge(u, v):
                        progress = True
                        break
                
                if progress:
                    break
            
            if not progress:
                break

        degs = np.sum(adj, axis=1)
        md = int(np.min(degs))
        
        if md >= n - 39:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        return None
    
    best_result = None
    
    for n in [47, 46, 45, 44]:
        if time.time() > deadline - 3:
            break
        
        for seed in range(100):
            if time.time() > deadline - 3:
                break
            result = solve(n, seed)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = {'n': 39, 'edges': []}
    
    return best_result
