import time
from collections import deque
import numpy as np
import sys

# Increase recursion limit for B&B
sys.setrecursionlimit(2000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute common neighbors
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c

    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin

    # Hopcroft-Karp for bipartite matching
    def hopcroft_karp(adj_left, n_left, n_right):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] -1
            found = False
            while q:
                u = q.popleft()
                for v in adj_left[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1
        return pair_u, matching

    def check_feasibility(B):
        """
        Checks if B is a valid set of branches using length-2 paths for non-adj pairs.
        Returns paths if feasible, else None.
        """
        B_set = set(B)
        non_adj = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj.append((u, v))
        
        if not non_adj:
            # All adjacent, just use direct edges
            paths = []
            for u, v in non_adj: # Empty
                pass
            for i in range(len(B)):
                for j in range(i + 1, len(B)):
                    paths.append([B[i], B[j]])
            return paths

        m = len(non_adj)
        right_verts = [v for v in range(n) if v not in B_set]
        right_idx = {v: i for i, v in enumerate(right_verts)}
        
        adj_left = []
        for u, v in non_adj:
            ws = [w for w in common[u][v] if w not in B_set]
            adj_left.append([right_idx[w] for w in ws if w in right_idx])
            
        if any(len(a) == 0 for a in adj_left):
            return None

        pair_u, matching = hopcroft_karp(adj_left, m, len(right_verts))
        
        if matching < m:
            return None

        # Construct paths
        paths = []
        # Direct edges for adjacent pairs
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v in adj[u]:
                    paths.append([u, v])
        
        # Non-adj paths
        for i, (u, v) in enumerate(non_adj):
            w_idx = pair_u[i]
            w = right_verts[w_idx]
            paths.append([u, w, v])
            
        return paths

    # Branch and Bound
    # Order vertices by degree descending to find large cliques/subdivisions early
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    best_B = []
    best_paths = None
    best_size = 0
    
    # Initial guess: Max Clique (fast to find, provides a lower bound)
    # Simple greedy clique
    clique = []
    remaining = set(range(n))
    while remaining:
        # Pick vertex with max degree in remaining
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        clique.append(v)
        remaining = remaining & adj[v]
    
    if clique:
        paths = check_feasibility(clique)
        if paths:
            best_B = clique
            best_paths = paths
            best_size = len(clique)
            
    # B&B
    def bb(idx, current_B):
        nonlocal best_B, best_paths, best_size
        
        if time.time() > deadline:
            return

        # Pruning
        if len(current_B) + (n - idx) <= best_size:
            return

        if idx == n:
            if len(current_B) > best_size:
                paths = check_feasibility(current_B)
                if paths:
                    best_B = list(current_B)
                    best_paths = paths
                    best_size = len(current_B)
            return

        v = order[idx]
        
        # 1. Include v
        new_B = current_B + [v]
        # Quick check: is new_B feasible?
        # To speed up, we can just call check_feasibility. 
        # If it's feasible, we recurse.
        if check_feasibility(new_B):
            bb(idx + 1, new_B)
            
        # 2. Exclude v
        bb(idx + 1, current_B)

    bb(0, [])
    
    # If B&B didn't find anything (or timed out), fall back to best known
    if not best_B:
        # Fallback: find any edge
        u, v = P['edges'][0]
        best_B = [u, v]
        best_paths = [[u, v]]

    # Final verification and type conversion
    final_paths = check_feasibility(best_B)
    if final_paths is None:
        # This should not happen if we tracked best_paths correctly, 
        # but just in case, revert to a known good state or simple edge
        if len(best_B) > 2:
             # Try to shrink? Or just return the last known good one?
             # The logic above updates best_B only if paths are found.
             pass
        else:
             u, v = P['edges'][0]
             best_B = [u, v]
             final_paths = [[u, v]]

    return {'branches': [int(b) for b in best_B], 'paths': [[int(x) for x in p] for p in final_paths]}

# EVOLVE-BLOCK-END
