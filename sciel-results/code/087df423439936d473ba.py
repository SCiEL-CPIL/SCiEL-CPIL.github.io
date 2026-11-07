import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths
    D = [[n] * n for _ in range(n)]
    for v in range(n):
        D[v][v] = 0
        q = deque([v])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[v][w] > D[v][u] + 1:
                    D[v][w] = D[v][u] + 1
                    q.append(w)

    def find_paths(src, dst, forbidden, max_paths=10):
        if src == dst: return []
        if dst in forbidden: return []
        
        dist = {src: 0}
        q = deque([src])
        parent = {src: None}
        
        while q:
            u = q.popleft()
            if u == dst: break
            for w in adj[u]:
                if w in forbidden: continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
        
        if dst not in dist: return []
        
        target_len = dist[dst]
        paths = []
        
        def dfs(u, path):
            if len(paths) >= max_paths: return
            if u == dst:
                paths.append(list(path))
                return
            for w in sorted(adj[u]):
                if w in forbidden: continue
                if dist.get(w, -1) == dist[u] + 1:
                    path.append(w)
                    dfs(w, path)
                    path.pop()
                    
        dfs(src, [src])
        return paths

    def check_branches(branches, time_limit=2.0):
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by distance
        pairs.sort(key=lambda p: D[p[0]][p[1]])
        
        used = set()
        paths = {}
        start_time = time.time()
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forb = branch_set - {u, v} | used
            candidates = find_paths(u, v, forb)
            
            if not candidates:
                return False
            
            # Shuffle to avoid bias? Or keep deterministic?
            # Deterministic is safer for reproducibility, but random might escape local minima.
            # Let's keep sorted (deterministic) for now.
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & used:
                    continue
                if interior & branch_set:
                    continue
                
                paths[(u, v)] = path
                used.update(interior)
                
                if backtrack(idx + 1):
                    return True
                
                used.difference_update(interior)
                del paths[(u, v)]
            
            return False

        if backtrack(0):
            return {
                'branches': branches,
                'paths': [[int(x) for x in paths[p]] for p in pairs]
            }
        return None

    # 1. Find Max Clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())
    
    best_result = None
    best_size = 0
    
    # Check Max Clique
    if len(best_clique) >= 2:
        res = check_branches(sorted(best_clique), time_limit=5.0)
        if res:
            best_result = res
            best_size = len(best_clique)

    # Try to expand
    degrees = [len(adj[i]) for i in range(n)]
    high_deg = sorted(range(n), key=lambda x: -degrees[x])
    pool = high_deg[:20]
    
    # Try sizes larger than max clique
    max_k_to_try = min(n, best_size + 3)
    
    for k in range(max_k_to_try, best_size, -1):
        if time.time() > deadline - 5:
            break
        if k <= best_size:
            continue
            
        # Strategy 1: Expand Max Clique
        if len(best_clique) >= 2:
            extras_needed = k - len(best_clique)
            if extras_needed > 0 and extras_needed <= 3: # Don't add too many at once
                available = [v for v in pool if v not in best_clique]
                if len(available) >= extras_needed:
                    for combo in combinations(available, extras_needed):
                        if time.time() > deadline - 2:
                            break
                        cand = sorted(list(best_clique) + list(combo))
                        res = check_branches(cand, time_limit=1.0)
                        if res:
                            return res

        # Strategy 2: Random from Pool
        if time.time() < deadline - 10:
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                cand = sorted(random.sample(pool, k))
                res = check_branches(cand, time_limit=0.5)
                if res:
                    return res

    if best_result:
        return best_result
        
    # Fallback
    if len(best_clique) >= 2:
        return {'branches': sorted(best_clique), 'paths': [[int(u), int(v)] for u, v in combinations(sorted(best_clique), 2) if u in adj[v]]}
        
    return {'branches': [0, 1], 'paths': [[0, 1]]}
