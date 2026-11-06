import numpy as np
import random

def construct():
    n = 31
    target_edges = 90
    
    def try_greedy(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        # Shuffle vertex pairs to break symmetry
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if len(edges) >= target_edges:
                break
            if adj[u][v]:
                continue
            
            # Check if adding (u,v) creates a C4
            # C4 exists if there is an edge between N(u) and N(v)
            creates_c4 = False
            for a in range(n):
                if adj[u][a]:
                    for b in range(n):
                        if a != b and adj[v][b] and adj[a][b]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
            
            if not creates_c4:
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
                
        return edges, adj

    best_edges = None
    best_adj = None
    min_score = float('inf')
    
    # Try multiple random seeds
    for s in range(100):
        edges, adj = try_greedy(s)
        if len(edges) == target_edges:
            # Find the best non-edge to add
            min_added = float('inf')
            best_non_edge = None
            
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        # Count C4s added
                        count = 0
                        for a in range(n):
                            if adj[i][a]:
                                for b in range(n):
                                    if a != b and adj[j][b] and adj[a][b]:
                                        count += 1
                        if count < min_added:
                            min_added = count
                            best_non_edge = [i, j]
            
            if best_non_edge is not None:
                if min_added < min_score:
                    min_score = min_added
                    best_edges = edges + [best_non_edge]
                    best_adj = adj
                    
                    # If we found 0, we can't do better
                    if min_score == 0:
                        return best_edges
                        
    if best_edges is not None:
        return best_edges
        
    # Fallback: return current polarity graph logic if greedy fails (unlikely)
    # But greedy should work.
    return []

