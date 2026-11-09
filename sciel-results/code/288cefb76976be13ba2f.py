"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u, v), max(u, v)))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(added_edges):
        adj = build_adj(added_edges)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def get_diameter(added_edges):
        adj = build_adj(added_edges)
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            if -1 in dist:
                return -1  # not connected
            max_diam = max(max_diam, max(dist))
        return max_diam
    
    def is_valid(added_edges):
        added_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in base_edges:
                return False
            if key in added_set:
                return False
            added_set.add(key)
        
        if not is_triangle_free(added_edges):
            return False
        
        diam = get_diameter(added_edges)
        if diam < 0 or diam > 4:
            return False
        
        return True
    
    # Greedy construction: start empty, add edges that reduce diameter most
    # while maintaining validity
    
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
    
    # Start with a good initial guess: edges of length 5
    current = []
    for i in range(n - 5):
        current.append((i, i + 5))
    
    if not is_valid(current):
        current = []
    
    # Try to find a valid solution with fewer edges
    # Use local search: start from a valid solution, try removing edges
    
    def try_remove_edges(added_edges):
        """Try to remove edges while maintaining validity"""
        removed = []
        changed = True
        while changed:
            changed = False
            for i in range(len(added_edges)):
                trial = added_edges[:i] + added_edges[i+1:]
                if is_valid(trial):
                    added_edges = trial
                    changed = True
                    break
        return added_edges
    
    if is_valid(current):
        current = try_remove_edges(current)
        return [[int(u), int(v)] for u, v in current]
    
    # If initial guess doesn't work, try different jump sizes
    for jump in [4, 6, 7, 8]:
        candidate = []
        for i in range(n - jump):
            candidate.append((i, i + jump))
        if is_valid(candidate):
            candidate = try_remove_edges(candidate)
            return [[int(u), int(v)] for u, v in candidate]
    
    # Try mixed jumps
    for j1 in [4, 5, 6]:
        for j2 in [7, 8, 9]:
            if j1 == j2:
                continue
            candidate = []
            for i in range(n - j1):
                candidate.append((i, i + j1))
            for i in range(n - j2):
                if (i, i + j2) not in candidate:
                    candidate.append((i, i + j2))
            if is_valid(candidate):
                candidate = try_remove_edges(candidate)
                return [[int(u), int(v)] for u, v in candidate]
    
    # Fallback: greedy add edges
    current = []
    for u, v in all_possible:
        trial = current + [(u, v)]
        if is_valid(trial):
            current = trial
    
    if is_valid(current):
        current = try_remove_edges(current)
        return [[int(u), int(v)] for u, v in current]
    
    # Last resort: return empty (will be invalid but better than crashing)
    return []
# EVOLVE-BLOCK-END
