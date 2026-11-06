import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_safe(N, u, v):
        if N[u] & (1 << v):
            return True
        
        C = N[u] & N[v]
        if popcount(C) < 2:
            return True
            
        c_list = [i for i in range(n) if (C >> i) & 1]
        for i in range(len(c_list)):
            for j in range(i+1, len(c_list)):
                v1 = c_list[i]
                v2 = c_list[j]
                S = N[v] & N[v1] & N[v2]
                S &= ~(1 << u)
                S &= ~(1 << v1)
                S &= ~(1 << v2)
                if popcount(S) >= 2:
                    return False
        return True

    best_edges = 0
    best_graph = None
    edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Run greedy multiple times
    while time.time() < deadline:
        random.shuffle(edges)
        N = [0] * n
        edge_count = 0
        
        for u, v in edges:
            if is_safe(N, u, v):
                N[u] |= (1 << v)
                N[v] |= (1 << u)
                edge_count += 1
                
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = [N[i] for i in range(n)]
            
    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u] & (1 << v):
                res.append([u, v])
    return res
