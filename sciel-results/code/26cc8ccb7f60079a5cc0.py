import time
import itertools
import sys
import random
import collections

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def has_edge(u, v):
        return (u, v) in edge_set

    # Precompute shortest paths between all pairs
    # Since n is small, we can run BFS for each node
    shortest_path = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        # BFS from i
        parent = [-1] * n
        dist = [-1] * n
        dist[i] = 0
        queue = collections.deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        for j in range(n):
            if i != j and dist[j] != -1:
                # Reconstruct path
                path = []
                cur = j
                while cur != -1:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                shortest_path[i][j] = path

    # Helper to find candidate paths for a pair (u, v)
    # We prefer shorter paths to reduce internal vertex conflicts
    # We will generate paths up to a certain length or just use the shortest one + some variations
    # For now, let's just use the shortest path as the primary candidate.
    # If that fails, we might need longer paths. 
    # Given the density, shortest paths are likely sufficient for small k.
    # But to be safe, let's allow the solver to try a few.
    
    def get_candidates(u, v, max_len=5):
        # Return list of paths, sorted by length
        # Use DFS to find simple paths
        paths = []
        def dfs(current, target, visited, path_so_far):
            if len(path_so_far) - 1 > max_len:
                return
            if current == target and len(path_so_far) > 1:
                paths.append(list(path_so_far))
                return
            for nb in adj[current]:
                if nb not in visited:
                    visited.add(nb)
                    path_so_far.append(nb)
                    dfs(nb, target, visited, path_so_far)
                    path_so_far.pop()
                    visited.remove(nb)
        
        visited = {u}
        dfs(u, v, visited, [u])
        paths.sort(key=lambda x: len(x))
        return paths

    # Solver for a fixed set of branches
    def solve_for_branches(branches):
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(itertools.combinations(branches, 2))
        pair_paths = {}
        for p in pairs:
            # Limit candidates to avoid explosion, shortest first
            cands = get_candidates(p[0], p[1], max_len=4)
            if not cands:
                return None
            pair_paths[p] = cands
        
        # Order pairs by number of options (ascending)
        pairs.sort(key=lambda p: len(pair_paths[p]))
        
        assigned_paths = {}
        used_internal_vertices = set()
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            p = pairs[idx]
            for path in pair_paths[p]:
                internals = set(path[1:-1])
                if internals & used_internal_vertices:
                    continue
                if internals & set(branches):
                    continue
                
                assigned_paths[p] = path
                used_internal_vertices.update(internals)
                
                if backtrack(idx + 1):
                    return True
                
                used_internal_vertices.difference_update(internals)
                del assigned_paths[p]
            
            return False
        
        if backtrack(0):
            return assigned_paths
        return None

    # Strategy:
    # 1. Find Max Clique
    def max_clique():
        best = []
        vertices = list(range(n))
        
        def expand(clique, candidates):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not candidates:
                return
            for i, v in enumerate(candidates):
                new_clique = clique + [v]
                new_candidates = [c for c in candidates[i+1:] if has_edge(v, c)]
                expand(new_clique, new_candidates)
        
        expand([], vertices)
        return best

    mc = max_clique()
    if len(mc) >= 2:
        branches = mc
        pairs = list(itertools.combinations(branches, 2))
        paths = {}
        for u, v in pairs:
            paths[(u,v)] = [u, v]
        # Verify (should be valid)
        return {
            'branches': [int(b) for b in branches],
            'paths': [list(map(int, paths[p])) for p in pairs]
        }

    # 2. Try to find larger k
    # We will try k from len(mc)+1 up to n
    # To speed up, we use randomized sampling of branch sets
    
    deadline = time.time() + 80 # 80s safety margin
    
    best_result = None
    best_k = len(mc)
    
    # Candidate vertices: All vertices, but prioritize high degree
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # We will try to find a set of size k
    # Start from k = best_k + 1
    for k in range(best_k + 1, n + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of trials for this k
        # If k is large, it's harder, so maybe fewer trials? Or more?
        # Let's do a fixed number of random trials per k, decreasing as k increases
        max_trials = 500 if k <= 8 else 100
        
        for _ in range(max_trials):
            if time.time() > deadline:
                break
                
            # Randomly select k vertices
            # Bias towards high degree vertices
            # Pick first 2 from top 10, rest random from all?
            # Or just random from all.
            branches = random.sample(sorted_vertices, k)
            
            # Quick check: Are any pair disconnected?
            # (Shortest path check is implicit in get_candidates)
            
            sol = solve_for_branches(branches)
            if sol:
                found = True
                best_k = k
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': [list(map(int, sol[p])) for p in itertools.combinations(branches, 2)]
                }
                break # Found for this k, try larger k
                
    if best_result:
        return best_result
        
    # Fallback to max clique result
    return {
        'branches': [int(b) for b in mc],
        'paths': [list(map(int, [u, v])) for u, v in itertools.combinations(mc, 2)]
    }
# EVOLVE-BLOCK-END
