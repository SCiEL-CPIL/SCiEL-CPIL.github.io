import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. LUW Construction
    # ... (same as before) ...
    
    # 2. Heuristic Search
    def check_girth_fast(adj, n, g):
        for s in range(n):
            visited = set()
            q = deque()
            for t in adj[s]:
                q.append((s, t, 1))
                visited.add((s, t))
            while q:
                u, prev, dist = q.popleft()
                if dist >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and dist + 1 >= 3:
                        if dist + 1 < g:
                            return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        q.append((v, u, dist + 1))
        return True

    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        if n < k + 1:
            return None
        for _ in range(10):
            half_edges = list(range(n * k))
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            edges_set = set()
            valid = True
            for i in range(0, n * k, 2):
                u = half_edges[i] // k
                v = half_edges[i + 1] // k
                if u == v or (u, v) in edges_set or (v, u) in edges_set:
                    valid = False
                    break
                edges_set.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            if valid and all(len(adj[i]) == k for i in range(n)):
                return adj
        return None

    def improve_girth(adj, n, k, g, time_limit):
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if check_girth_fast(adj, n, g):
                return True
            # Find a short cycle
            # For simplicity, pick a random edge and try to swap it
            u = random.randint(0, n-1)
            if not adj[u]:
                continue
            v = random.choice(adj[u])
            
            # Remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find (w,x) to swap
            w, x = -1, -1
            for _ in range(50):
                w = random.randint(0, n-1)
                if w == u or w == v:
                    continue
                if not adj[w]:
                    continue
                x = random.choice(adj[w])
                if x == u or x == v:
                    continue
                if w in adj[u] or x in adj[v]:
                    continue
                if u == w or v == x:
                    continue
                break
            else:
                # Revert
                adj[u].append(v)
                adj[v].append(u)
                continue
                
            # Swap
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
        return False

    # Try sizes
    for target_n in [2000, 1500, 1200]:
        if time.time() > deadline - 10:
            break
        adj = random_regular_graph(target_n, k)
        if adj:
            if improve_girth(adj, target_n, k, g, 5):
                edges = [(u, v) for u in range(target_n) for v in adj[u] if u < v]
                return target_n, edges

    return best_n, best_edges
