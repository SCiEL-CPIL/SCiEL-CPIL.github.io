import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    if k != 3:
        # Fallback for other k
        return None, None
        
    # We need a 3-regular graph with girth >= g (16).
    # Start with n = 512.
    
    def gen_3_reg(n):
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        edges = set()
        for i in range(0, n, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v: return None
            if (u, v) in edges or (v, u) in edges: return None
            edges.add((u, v))
        return list(edges)

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                v = q.popleft()
                if dist[v] >= g // 2 - 1: continue
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        prev[u] = v
                        q.append(u)
                    else:
                        if u != prev[v] and v != prev[u]:
                            if dist[v] + dist[u] + 1 < g:
                                return False
        return True

    n = 512
    best_n, best_edges = None, None
    
    while time.time() < deadline:
        edges = gen_3_reg(n)
        if edges:
            if check_girth(n, edges, g):
                return n, edges
        n += 2
        
    return best_n, best_edges
