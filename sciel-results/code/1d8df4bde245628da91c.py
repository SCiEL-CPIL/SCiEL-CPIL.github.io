import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80
    
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    def is_valid(added):
        adj = [set(s) for s in base_adj]
        for u, v in added:
            if u in adj[v]:
                return False
            if len(adj[u] & adj[v]) > 0:
                return False
            adj[u].add(v)
            adj[v].add(u)
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
        return True

    best_solution = None

    for num_edges in [6, 7]:
        if best_solution is not None:
            break
        if time.time() > deadline:
            break
            
        for c in range(n):
            if time.time() > deadline:
                break
            
            candidates = [v for v in range(n) if v != c and v not in base_adj[c]]
            m = len(candidates)
            
            # Build incompatibility sets
            incomp = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i+1, m):
                    if candidates[j] in base_adj[candidates[i]]:
                        incomp[i].add(j)
                        incomp[j].add(i)
            
            # Generate independent sets of size num_edges with pruning
            def gen_sets(size, current, start):
                if len(current) == size:
                    yield tuple(current)
                    return
                remaining = size - len(current)
                if m - start < remaining:
                    return
                for i in range(start, m):
                    if any(i in incomp[j] for j in current):
                        continue
                    current.append(i)
                    yield from gen_sets(size, current, i+1)
                    current.pop()
            
            for combo_idx in gen_sets(num_edges, [], 0):
                if time.time() > deadline:
                    break
                combo = [candidates[i] for i in combo_idx]
                edges = [(c, v) for v in combo]
                if is_valid(edges):
                    best_solution = edges
                    break
            if best_solution:
                break
    
    if best_solution is None:
        return []

    return [[int(u), int(v)] for u, v in best_solution]
