import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Precompute which pairs can be added
    can_add = [[False] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if v not in base_adj[u]:
                can_add[u][v] = True
                can_add[v][u] = True
    
    # Precompute common neighbors for triangle check
    common_neighbors = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v:
                common_neighbors[u][v] = base_adj[u] & base_adj[v]
    
    def check_diameter(adj):
        """Check if diameter <= 4. Returns True if valid."""
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
            if any(d == -1 for d in dist):
                return False
        return True
    
    best_solution = None
    best_size = 100
    
    # For each potential hub, find minimum set of vertices to connect
    # such that all vertices are within distance 2 of the hub
    for c in range(n):
        if time.time() > deadline:
            break
        
        # Get valid candidates for hub c
        valid_cands = []
        for v in range(n):
            if v == c or not can_add[c][v]:
                continue
            if len(common_neighbors[c][v]) > 0:
                continue
            valid_cands.append(v)
        
        if not valid_cands:
            continue
        
        # Greedy set cover: each candidate v covers {v-1, v, v+1} intersect [0, n-1]
        # Plus c is always covered
        uncovered = set(range(n)) - {c}
        
        # Sort candidates by coverage (number of uncovered vertices they cover)
        def coverage(v):
            cov = set()
            for x in [v - 1, v, v + 1]:
                if 0 <= x < n and x in uncovered:
                    cov.add(x)
            return len(cov)
        
        chosen = []
        remaining_uncovered = set(uncovered)
        
        # Greedy selection
        while remaining_uncovered:
            # Find candidate with max coverage of remaining_uncovered
            best_v = None
            best_cov = -1
            for v in valid_cands:
                if v in chosen:
                    continue
                cov = 0
                for x in [v - 1, v, v + 1]:
                    if 0 <= x < n and x in remaining_uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            
            if best_v is None or best_cov == 0:
                break
            
            chosen.append(best_v)
            # Update remaining_uncovered
            for x in [best_v - 1, best_v, best_v + 1]:
                if 0 <= x < n:
                    remaining_uncovered.discard(x)
        
        # Now try to verify and possibly improve
        # Check if chosen is valid (independent set, no triangles)
        def is_independent(cands):
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    if cands[j] in base_adj[cands[i]]:
                        return False
            return True
        
        if is_independent(chosen):
            adj = [set(s) for s in base_adj]
            for v in chosen:
                adj[c].add(v)
                adj[v].add(c)
            
            if check_diameter(adj):
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_solution = [[c, v] for v in chosen]
        
        # Try local search: swap one vertex for another
        if len(chosen) <= 8:
            for _ in range(50):  # limited iterations
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(chosen)):
                    for v in valid_cands:
                        if v in chosen:
                            continue
                        new_chosen = chosen[:i] + [v] + chosen[i + 1:]
                        if not is_independent(new_chosen):
                            continue
                        adj = [set(s) for s in base_adj]
                        for w in new_chosen:
                            adj[c].add(w)
                            adj[w].add(c)
                        if check_diameter(adj):
                            # Check coverage
                            covered = {c}
                            for w in new_chosen:
                                for x in [w - 1, w, w + 1]:
                                    if 0 <= x < n:
                                        covered.add(x)
                            if covered == set(range(n)):
                                chosen = new_chosen
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
            
            if is_independent(chosen):
                adj = [set(s) for s in base_adj]
                for v in chosen:
                    adj[c].add(v)
                    adj[v].add(c)
                if check_diameter(adj):
                    covered = {c}
                    for v in chosen:
                        for x in [v - 1, v, v + 1]:
                            if 0 <= x < n:
                                covered.add(x)
                    if covered == set(range(n)) and len(chosen) < best_size:
                        best_size = len(chosen)
                        best_solution = [[c, v] for v in chosen]
    
    if best_solution is None:
        return []
    
    return [[int(u), int(v)] for u, v in best_solution]
