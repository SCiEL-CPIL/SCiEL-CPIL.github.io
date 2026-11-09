# EVOLVE-BLOCK-START
import itertools
from collections import deque
import random

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
                return -1
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

    def local_search_minimize(added_edges):
        current = list(added_edges)
        if not is_valid(current):
            return None
            
        improved = True
        while improved:
            improved = False
            indices = list(range(len(current)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(current):
                    continue
                trial = current[:i] + current[i+1:]
                if is_valid(trial):
                    current = trial
                    improved = True
                    break
        return current

    best_solution = None

    # Try hub-based configurations
    # Pick a center, connect to spread out nodes
    for center in [10, 11, 12, 13, 14]:
        # Generate candidate neighbors: nodes at distance 3-8 from center
        candidates = []
        for d in [3, 4, 5, 6, 7, 8]:
            candidates.append((center - d, center))
            candidates.append((center + d, center))
        
        # Filter valid candidates (within bounds, not base edge)
        valid_cands = []
        for u, v in candidates:
            if 0 <= u < n and 0 <= v < n:
                key = (min(u, v), max(u, v))
                if key not in base_edges:
                    valid_cands.append(key)
        
        # Greedily pick edges to minimize diameter
        # Start with empty set, add edges that reduce max distance most
        # Or just try small combinations
        
        # Try combinations of size 4, 5, 6
        for k in [4, 5, 6]:
            # Random sampling of combinations
            for _ in range(1000):
                if len(valid_cands) < k:
                    continue
                sample = random.sample(valid_cands, k)
                if is_valid(sample):
                    minimized = local_search_minimize(sample)
                    if minimized is not None:
                        if best_solution is None or len(minimized) < len(best_solution):
                            best_solution = minimized

    # Fallback to original heuristic if no hub solution found
    if best_solution is None:
        for jump in [5, 6, 7, 8]:
            candidate = []
            for i in range(n - jump):
                candidate.append((i, i + jump))
            if is_valid(candidate):
                minimized = local_search_minimize(candidate)
                if minimized is not None:
                    if best_solution is None or len(minimized) < len(best_solution):
                        best_solution = minimized

    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    return []
# EVOLVE-BLOCK-END
