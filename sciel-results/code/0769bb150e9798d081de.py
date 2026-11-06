import random
import time

def construct():
    n = 20
    best_edges = []
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [[False]*n for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if adj[u][v]:
                continue
            
            # Get current neighbors of u and v
            # We only need to check w that are neighbors of v (for u,w pair)
            # and w that are neighbors of u (for v,w pair)
            
            valid = True
            
            # Check pairs (u, w) where w is neighbor of v
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[w][v]:
                    if common[u][w] + 1 > 2:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check pairs (v, w) where w is neighbor of u
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[w][u]:
                    if common[v][w] + 1 > 2:
                        valid = False
                        break
            
            if valid:
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append([u, v])
                
                # Update common counts
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[w][v]:
                        common[u][w] += 1
                        common[w][u] += 1
                    if adj[w][u]:
                        common[v][w] += 1
                        common[w][v] += 1
                        
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            # Optional: if we reach a very high number, we could break, 
            # but let's run until deadline to be safe.
            
    return best_edges
