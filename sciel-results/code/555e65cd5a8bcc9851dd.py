import random
import numpy as np

def construct():
    n = 40
    best_edges = []
    best_count = 0
    
    for restart in range(20):
        adj = np.zeros((n, n), dtype=int)
        
        def add_edge(u, v):
            adj[u, v] = 1
            adj[v, u] = 1

        def remove_edge(u, v):
            adj[u, v] = 0
            adj[v, u] = 0

        def has_c6(u, v):
            # Check for path of length 5
            # BFS is often faster than DFS for shortest path, 
            # but we need ANY path of length 5.
            # If shortest path < 5, there might still be a path of length 5.
            # However, in a dense graph, if shortest path is 2, 
            # there is likely a path of length 5.
            # Let's use DFS with pruning.
            
            # Optimization: If u and v are not connected by a path of length 5,
            # it's because the graph is sparse or structured.
            
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, mask = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                
                neighbors = np.where(adj[node] == 1)[0]
                for nb in neighbors:
                    if not (mask & (1 << nb)):
                        stack.append((nb, depth + 1, mask | (1 << nb)))
            return False

        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        added = set()
        
        for u, v in all_edges:
            if not has_c6(u, v):
                add_edge(u, v)
                added.add((u, v))
        
        count = len(added)
        if count > best_count:
            best_count = count
            best_edges = list(added)
            
    return [[u, v] for u, v in best_edges]
