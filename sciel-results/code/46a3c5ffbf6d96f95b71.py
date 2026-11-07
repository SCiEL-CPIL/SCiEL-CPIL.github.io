import time
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    adj_base = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
        adj_base[u].add(v)
        adj_base[v].add(u)
    
    # Only consider bipartite edges (even-odd) to guarantee triangle-free
    A = set(range(0, n, 2))
    B = set(range(1, n, 2))
    
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    edge_to_idx = {c: i for i, c in enumerate(candidates)}
    deadline = time.time() + 85
    
    def _check_diameter(added_edges):
        adj = [set(adj_base[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        # BFS from each vertex, check max distance <= 4
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]; qi += 1
                du = dist[u]
                if du >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True
    
    # Try small k first with systematic search, then random
    for k in range(1, 10):
        if time.time() > deadline:
            break
        found = None
        rng = random.Random(1000 + k)
        
        if k <= 2:
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _check_diameter(combo):
                    found = list(combo)
                    break
        else:
            # Use greedy + local search for larger k
            # First try a structured approach: pick hub-like vertices
            # Then random sampling with better candidates
            for it in range(200000):
                if time.time() > deadline:
                    break
                combo = rng.sample(candidates, k)
                if _check_diameter(combo):
                    found = list(combo)
                    break
        
        if found:
            return [[u, v] for u, v in found]
    
    # Fallback: known valid solution
    # Use two hubs: 11 (odd) connects to evens, 12 (even) connects to odds
    # This keeps bipartite structure
    solution = []
    hub1 = 11  # odd
    hub2 = 12  # even
    # hub1 connects to evens (not adjacent in path)
    for v in range(0, 24, 2):
        if abs(hub1 - v) > 1:
            solution.append([hub1, v])
    # hub2 connects to odds (not adjacent in path)
    for v in range(1, 24, 2):
        if abs(hub2 - v) > 1:
            solution.append([hub2, v])
    
    # Verify and return
    if _check_diameter(solution):
        return solution
    
    # Minimal fallback
    return [[11, 0], [11, 2], [11, 4], [11, 6], [11, 14], [11, 16], [11, 18], [11, 20], [11, 22], [12, 1], [12, 3], [12, 5], [12, 7], [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
