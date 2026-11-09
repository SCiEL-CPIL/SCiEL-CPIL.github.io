import random
import time

def construct():
    n = 20
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        current_edges = []
        
        for u, v in edges_to_try:
            if common[u][v] > 2:
                continue
            
            valid = True
            for w in adj[u]:
                if common[v][w] > 1:
                    valid = False
                    break
            
            if valid:
                for w in adj[v]:
                    if common[u][w] > 1:
                        valid = False
                        break
            
            if valid:
                old_Nu = list(adj[u])
                old_Nv = list(adj[v])
                
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                
                for w in old_Nu:
                    common[v][w] += 1
                    common[w][v] += 1
                for w in old_Nv:
                    common[u][w] += 1
                    common[w][u] += 1
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    return best_edges
