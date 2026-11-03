import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def build_tensor_product(L, M):
        n = L * M
        edges = set()
        for i in range(L):
            for j in range(M):
                v = i * M + j
                for di in [-1, 1]:
                    for dj in [-1, 1]:
                        ni = (i + di) % L
                        nj = (j + dj) % M
                        nv = ni * M + nj
                        if v < nv:
                            edges.add((v, nv))
        return n, list(edges)

    def compute_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth <= g:
                break
        return min_girth

    candidates = []
    for L in range(10, 40):
        for M in range(10, 40):
            if L * M >= 161 and L * M <= 600:
                candidates.append((L, M))
                
    candidates.sort(key=lambda x: x[0] * x[1])
    
    best_result = None
    best_n = float('inf')
    
    for L, M in candidates:
        if time.time() > deadline - 10:
            break
            
        n, edges = build_tensor_product(L, M)
        
        if n < best_n:
            girth = compute_girth(n, edges)
            if girth >= g:
                best_n = n
                best_result = (n, edges)
                
    if best_result is not None:
        return best_result
        
    return (0, [])
