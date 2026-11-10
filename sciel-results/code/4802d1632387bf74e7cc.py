"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_paths_bfs(adj, src, dst, forbidden, max_paths=50):
    """Find up to max_paths simple paths from src to dst avoiding forbidden interior vertices."""
    if src == dst:
        return [[src]]
    if src in forbidden or dst in forbidden:
        return []
    
    all_paths = []
    # DFS with backtracking to find multiple paths
    def dfs(u, path, visited):
        if len(all_paths) >= max_paths:
            return
        if u == dst:
            all_paths.append(list(path))
            return
        for v in adj[u]:
            if v not in visited and v not in forbidden:
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.remove(v)
                if len(all_paths) >= max_paths:
                    return
    
    dfs(src, [src], {src})
    return all_paths

def _try_assign_paths(adj, branches, deadline):
    """Try to assign vertex-disjoint interior paths for all pairs in branches."""
    pairs = list(combinations(range(len(branches)), 2))
    # Sort pairs: adjacent pairs first, then by number of available paths
    pair_info = []
    for i, j in pairs:
        bi, bj = branches[i], branches[j]
        if bi in adj[bj]:
            pair_info.append((0, i, j, [bi, bj]))
        else:
            pair_info.append((1, i, j, None))
    pair_info.sort(key=lambda x: x[0])
    
    n_pairs = len(pair_info)
    paths = [None] * n_pairs
    used_interior = set()
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n_pairs:
            return True
        _, i, j, precomputed = pair_info[idx]
        bi, bj = branches[i], branches[j]
        
        if precomputed is not None:
            # Direct edge
            if (bi, bj) in [(a, b) for a, b in []] or True:
                interior = set()
                paths[idx] = precomputed
                if backtrack(idx + 1):
                    return True
                paths[idx] = None
            return False
        
        forbidden = used_interior | set(branches) - {bi, bj}
        candidate_paths = _find_paths_bfs(adj, bi, bj, forbidden, max_paths=20)
        
        # Sort by length
        candidate_paths.sort(key=len)
        
        for cp in candidate_paths:
            interior = set(cp[1:-1])
            if interior & used_interior:
                continue
            used_interior.update(interior)
            paths[idx] = cp
            if backtrack(idx + 1):
                return True
            used_interior -= interior
            paths[idx] = None
        
        return False
    
    if backtrack(0):
        return paths
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    # Greedy coloring
    def greedy_color(candidates):
        colors = []
        color_sets = []
        for v in candidates:
            placed = False
            for ci, cs in enumerate(color_sets):
                if all(u != v for u in cs):
                    cs.append(v)
                    colors.append(ci)
                    placed = True
                    break
            if not placed:
                color_sets.append([v])
                colors.append(len(color_sets) - 1)
        return colors
    
    best_clique = []
    
    def expand_clique(candidates, clique):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            return
        colors = greedy_color(candidates)
        max_color = max(colors) + 1 if colors else 0
        if len(clique) + max_color <= len(best_clique):
            return
        for idx in range(len(candidates) - 1, -1, -1):
            v = candidates[idx]
            new_clique = clique + [v]
            new_candidates = [c for c in candidates[:idx] if c in adj[v]]
            expand_clique(new_candidates, new_clique)
            if time.time() > deadline:
                return
    
    expand_clique(list(range(n)), [])
    
    # Try the best clique and its subsets
    candidates_to_try = []
    if len(best_clique) >= 2:
        candidates_to_try.append(best_clique)
        # Try subsets of size len(best_clique)-1, etc.
        for size in range(len(best_clique) - 1, 1, -1):
            for subset in combinations(best_clique, size):
                candidates_to_try.append(list(subset))
    
    # Also try other large cliques if we found them
    # For now, just try the best and its subsets
    
    for cand in candidates_to_try:
        if time.time() > deadline:
            break
        if len(cand) < 2:
            continue
        paths = _try_assign_paths(adj, cand, deadline)
        if paths is not None:
            branch_list = list(cand)
            path_list = []
            for p in paths:
                path_list.append(list(p))
            return {'branches': branch_list, 'paths': path_list}
    
    # Fallback: any edge
    for u, v in P['edges']:
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
