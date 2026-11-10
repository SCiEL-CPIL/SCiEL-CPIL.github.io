"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = _time.time() + 55  # safety margin
    
    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    def check_branch_set(branches):
        """Check if a set of branches has valid disjoint paths. Returns paths dict or None."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        used_interiors = set()
        paths = {}
        
        # Sort pairs by shortest path length
        def pair_key(p):
            path = find_shortest_path(p[0], p[1], bset - {p[0], p[1]})
            if path is None:
                return (float('inf'), p)
            return (len(path), p)
        
        sorted_pairs = sorted(pairs, key=pair_key)
        
        for u, v in sorted_pairs:
            forbidden = (bset - {u, v}) | used_interiors
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path
        
        return paths
    
    def try_grow(initial_branches):
        """Try to grow a branch set greedily."""
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = check_branch_set(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current
    
    def find_max_clique():
        """Find a maximum clique using Bron-Kerbosch with pivot."""
        best_clique = []
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Choose pivot
            U = P | X
            pivot = max(U, key=lambda v: len(P & adj[v])) if U else None
            candidates = P - (adj[pivot] if pivot is not None else set())
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return best_clique
    
    # Try multiple starting points
    best_branches = None
    best_paths = None
    
    # Start with maximum clique
    clique = find_max_clique()
    if len(clique) >= 2:
        result = try_grow(clique)
        paths = check_branch_set(result)
        if paths is not None:
            best_branches = result
            best_paths = paths
    
    # Also try starting from top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:6]
    result = try_grow(list(top))
    paths = check_branch_set(result)
    if paths is not None and (best_branches is None or len(result) > len(best_branches)):
        best_branches = result
        best_paths = paths
    
    # Try random-ish starts
    import random
    random.seed(42)
    for _ in range(50):
        if _time.time() > deadline:
            break
        start = random.sample(range(n), min(3, n))
        result = try_grow(start)
        paths = check_branch_set(result)
        if paths is not None and (best_branches is None or len(result) > len(best_branches)):
            best_branches = result
            best_paths = paths
    
    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
    
    # Format output
    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
