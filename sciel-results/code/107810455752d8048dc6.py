import time
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

    # Precompute all-pairs shortest paths (distance only) for pruning
    # D[v][u] = shortest distance from v to u
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

    def find_paths(src, dst, forbidden_mask, max_len=None):
        """
        Find simple paths from src to dst avoiding vertices in forbidden_mask.
        Returns a list of paths (lists of vertices), sorted by length.
        """
        if src == dst:
            return []
        if dst in forbidden_mask:
            return []
        
        # BFS to find shortest distance
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for w in adj[u]:
                if w not in dist and w not in forbidden_mask:
                    dist[w] = dist[u] + 1
                    q.append(w)
        
        if dst not in dist:
            return []
        
        shortest_len = dist[dst]
        if max_len is None:
            max_len = shortest_len + 2  # Allow slightly longer paths
        
        paths = []
        
        def dfs(u, path, visited):
            if len(paths) > 30:  # Cap to avoid explosion
                return
            if u == dst:
                if len(path) - 1 <= max_len:
                    paths.append(list(path))
                return
            
            remaining = max_len - (len(path) - 1)
            if remaining <= 0:
                return
            
            # Prune: if shortest path from u to dst exceeds remaining, skip
            if u in dist and dst not in dist:
                return
            
            for w in sorted(adj[u]):
                if w in forbidden_mask or w in visited:
                    continue
                # Prune: don't go further than max_len
                if len(path) - 1 + 1 > max_len:
                    continue
                visited.add(w)
                path.append(w)
                dfs(w, path, visited)
                path.pop()
                visited.remove(w)
                if len(paths) > 30:
                    return
        
        dfs(src, [src], {src})
        
        # Sort by length (shortest first)
        paths.sort(key=len)
        return paths

    def solve_for_branches(branches, time_limit):
        """
        Attempt to find internally disjoint paths for all pairs in branches.
        Returns dict of paths or None.
        """
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # We will use backtracking.
        # State: index of pair, set of used interior vertices.
        
        paths = {}
        
        # Order pairs: those with fewer "options" first.
        # Use distance as a heuristic: shorter distance = fewer interior vertices = more constrained
        pairs.sort(key=lambda p: D[p[0]][p[1]])

        start_time = time.time()
        
        def backtrack(idx, used):
            if time.time() - start_time > time_limit:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forb = branch_set - {u, v} | used
            
            # Try shortest paths first, then slightly longer
            for max_len in [D[u][v], D[u][v] + 1, D[u][v] + 2]:
                if max_len > n:
                    break
                candidate_paths = find_paths(u, v, forb, max_len=max_len)
                
                if not candidate_paths:
                    continue
                
                # Try each candidate path
                for path in candidate_paths:
                    interior = set(path[1:-1])
                    if interior & used:
                        continue
                    if interior & branch_set:
                        continue
                    
                    paths[(u, v)] = path
                    if backtrack(idx + 1, used | interior):
                        return True
                    del paths[(u, v)]
                
                if paths:
                    # Found a path for this pair, but backtracking failed
                    # Try next max_len
                    continue
            
            return False

        if backtrack(0, set()):
            return paths
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
    
    # 2. Generate Candidates
    # High degree vertices
    high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Use a larger pool for more coverage
    top_k = 20
    pool = high_deg[:top_k]
    
    best_result = None
    best_size = 0
    
    # Try Max Clique first
    if best_clique and len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = solve_for_branches(branches, time_limit=3.0)
        if paths:
            best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
            best_size = len(branches)
    
    # Try larger sizes from pool
    # We go from large to small to find the maximum.
    # Start with size 12 down to 8
    for size in [12, 11, 10, 9, 8]:
        if time.time() > deadline - 10:
            break
        if size <= best_size:
            continue
            
        # Generate combos
        for combo in combinations(pool, size):
            if time.time() > deadline - 3:
                break
            branches = sorted(combo)
            # Quick check: are all pairs connected?
            skip = False
            for u, v in combinations(branches, 2):
                if D[u][v] == n:
                    skip = True
                    break
            if skip:
                continue
                
            # Use a tight time limit per attempt
            paths = solve_for_branches(branches, time_limit=0.3)
            if paths:
                best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
                best_size = size
                # We found one of this size. Since we're going from large to small,
                # this is the best we can find. Return immediately.
                return best_result
                
    # If we didn't return, return best_result or fallback
    if best_result is None:
        # Fallback: find any valid pair
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    return best_result
