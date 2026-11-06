import numpy as np
import time
import random

def count_c4(adj):
    n = len(adj)
    total = 0
    for u in range(n):
        for v in range(u+1, n):
            c = np.sum(adj[u] & adj[v])
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 110
    
    random.seed(42)
    
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for i, j in all_pairs[:target_edges]:
        adj[i, j] = adj[j, i] = True
        edges.add((i, j))
    
    best_c4 = count_c4(adj)
    best_adj = adj.copy()
    best_edges = set(edges)
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        edge_list = list(best_edges)
        if not edge_list:
            break
        u, v = random.choice(edge_list)
        
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                     if (i,j) not in best_edges]
        if not non_edges:
            break
        x, y = random.choice(non_edges)
        
        best_adj[u, v] = best_adj[v, u] = False
        best_edges.discard((u,v))
        best_adj[x, y] = best_adj[y, x] = True
        best_edges.add((x,y))
        
        c4_new = count_c4(best_adj)
        
        if c4_new < best_c4 or (c4_new - best_c4) < random.expovariate(1.0/max(temperature, 0.01)):
            best_c4 = c4_new
        else:
            best_adj[u, v] = best_adj[v, u] = True
            best_edges.add((u,v))
            best_adj[x, y] = best_adj[y, x] = False
            best_edges.discard((x,y))
        
        temperature *= cooling
    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
# EVOLVE-BLOCK-END
