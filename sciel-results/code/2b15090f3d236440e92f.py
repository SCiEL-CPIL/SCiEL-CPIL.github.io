# EVOLVE-BLOCK-START
import time
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    start_time = time.time()
    deadline = start_time + 80.0

    # Precompute common neighbours
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c

    # Max clique via Bron-Kerbosch with pivot
    def find_max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if Pset:
                pivot = max(Pset, key=lambda v: len(adj[v] & Pset))
                cands = Pset - adj[pivot]
            else:
                cands = Pset
            for v in list(cands):
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return best

    def check_feasibility(B):
        """
        Checks if there is a set of pairwise disjoint internal vertices for all non-adj pairs in B.
        Returns (paths) if feasible, else None.
        Uses Hopcroft-Karp matching.
        """
        B_set = set(B)
        non_adj = [(u, v) for u in B for v in B if u < v and v not in adj[u]]
        
        if len(non_adj) > n - len(B):
            return None

        cands = []
        for u, v in non_adj:
            ws = [w for w in common[u][v] if w not in B_set]
            cands.append(ws)
            if not ws:
                return None

        # Hopcroft-Karp Matching
        m = len(non_adj)
        right_verts = [v for v in range(n) if v not in B_set]
        right_idx = {v: i for i, v in enumerate(right_verts)}
        
        adj_left = []
        for ws in cands:
            adj_left.append([right_idx[w] for w in ws if w in right_idx])
            
        pair_u = [-1] * m
        pair_v = [-1] * len(right_verts)
        dist = [0] * m

        def bfs():
            q = deque()
            for u in range(m):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
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
            for u in range(m):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1

        if matching < m:
            return None

        # Construct paths
        paths = []
        # Direct edges
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v in adj[u]:
                    paths.append([u, v])
        
        # Non-adj paths
        for i, (u, v) in enumerate(non_adj):
            w = right_verts[pair_u[i]]
            paths.append([u, w, v])
            
        return paths

    # 1. Start with Max Clique
    clique = find_max_clique()
    if len(clique) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    B = list(clique)
    best_B = list(B)
    
    # 2. Greedy Expansion
    # Candidates ordered by "potential" (e.g., degree)
    candidates = sorted([v for v in range(n) if v not in B], key=lambda x: -degrees[x])
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for v in candidates:
            if v in B:
                continue
                
            # Try adding v
            new_B = B + [v]
            paths = check_feasibility(new_B)
            if paths is not None:
                B = new_B
                best_B = list(B)
                candidates.remove(v)
                improved = True
                break # Restart loop with new B
                
    # 3. Local Search / Swap (Optional, if time permits)
    # Try to swap a vertex in B with one outside to allow further growth?
    # For now, the greedy expansion is a strong heuristic.
    # We can also try removing a vertex and re-adding others if stuck, but for now keep it simple.
    
    # Final check and return
    final_paths = check_feasibility(B)
    if final_paths is None:
        # Should not happen if we started with a clique (which is always feasible)
        # and only added if feasible. But just in case.
        B = list(best_B)
        final_paths = check_feasibility(B)
        
    if len(B) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': [int(b) for b in B], 'paths': [[int(x) for x in p] for p in final_paths]}

# EVOLVE-BLOCK-END
