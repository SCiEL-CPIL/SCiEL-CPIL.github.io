"""Combined: Tomita-coloring Bron-Kerbosch + backtracking disjoint path search."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order

def _color_sort(adj, P_list):
    """Greedy coloring: returns (order, colors) where order is a list of vertices
    and colors[i] is the color of order[i].  Colors are 1..k with the property that
    any clique in P has size <= max(colors)."""
    uncolored = set(P_list)
    colors = []
    order = []
    color_class = []
    while uncolored:
        # pick a vertex for the next color class
        # greedy: pick vertex with max degree within uncolored
        v = max(uncolored, key=lambda x: len(adj[x] & uncolored))
        color_class.append(v)
        uncolored.discard(v)
        # remove all neighbors of v from uncolored (they can't share this color)
        uncolored -= adj[v]
        colors.append(len(colors) + 1)
        order.append(v)
    # Now we have a valid coloring.  Reorder so that higher colors come first
    # (Tomita ordering: sort by color descending for better pruning)
    pairs = list(zip(order, colors))
    pairs.sort(key=lambda x: -x[1])
    order = [p[0] for p in pairs]
    colors = [p[1] for p in pairs]
    return order, colors

def _expand(adj, R, P_list, best_clique, deadline):
    """Tomita-style expansion with coloring bound."""
    if time.time() > deadline:
        return best_clique
    if not P_list:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    # Color the candidate set
    order, colors = _color_sort(adj, P_list)
    # Process in reverse color order (highest color first)
    for i in range(len(order) - 1, -1, -1):
        if time.time() > deadline:
            return best_clique
        v = order[i]
        c = colors[i]
        # Upper bound: |R| + c
        if len(R) + c <= len(best_clique):
            break
        # Recurse
        new_R = R | {v}
        new_P = [w for w in P_list if w in adj[v]]
        best_clique = _expand(adj, new_R, new_P, best_clique, deadline)
        P_list.remove(v)
    return best_clique

def _max_clique(adj, deadline):
    n = P['n']
    order = _degeneracy_order(adj)
    best = []
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        P_list = [w for w in adj[v] if w in set(order[i+1:])]
        if not P_list:
            continue
        best = _expand(adj, {v}, P_list, best, deadline)
    if not best:
        best = _expand(adj, set(), list(range(n)), best, deadline)
    return best

def _find_shortest_paths(adj, src, dst, forbidden, max_paths, max_len):
    """Find up to max_paths shortest simple paths from src to dst avoiding forbidden interior vertices."""
    if src == dst:
        return []
    best_len = None
    found = []
    # BFS for shortest path length
    visited = {src: 0}
    queue = deque([(src, [src])])
    shortest = None
    while queue:
        node, path = queue.popleft()
        if node == dst:
            shortest = len(path)
            break
        for nb in adj[node]:
            if nb not in visited:
                if nb in forbidden and nb != src and nb != dst:
                    continue
                visited[nb] = len(path) + 1
                queue.append((nb, path + [nb]))
    if shortest is None:
        return []
    # DFS to find up to max_paths paths of length <= shortest + 2 (allow some longer ones)
    limit_len = min(shortest + 3, max_len)
    def dfs(node, path, used):
        if len(found) >= max_paths:
            return
        if node == dst:
            found.append(list(path))
            return
        if len(path) >= limit_len:
            return
        for nb in adj[node]:
            if nb in used:
                continue
            if nb in forbidden and nb != src and nb != dst:
                continue
            used.add(nb)
            path.append(nb)
            dfs(nb, path, used)
            path.pop()
            used.discard(nb)
            if len(found) >= max_paths:
                return
    dfs(src, [src], {src})
    return found

def _backtrack_paths(adj, branches, deadline):
    """
    Find internally vertex-disjoint paths for all pairs of branch vertices.
    Uses backtracking with multiple candidate paths per pair.
    Returns (success, paths_dict) where paths_dict maps (u,v) to path list.
    """
    n = P['n']
    k = len(branches)
    if k < 2:
        return False, {}
    
    branch_set = set(branches)
    sorted_branches = sorted(branches)
    pairs = list(combinations(sorted_branches, 2))
    
    # For each pair, find candidate paths (avoiding other branches as interior)
    pair_candidates = {}
    for u, v in pairs:
        forbidden = branch_set - {u, v}
        cands = _find_shortest_paths(adj, u, v, forbidden, max_paths=20, max_len=15)
        pair_candidates[(u, v)] = cands
        if not cands:
            return False, {}
    
    # Order pairs by number of candidates (ascending) for better backtracking
    pairs_ordered = sorted(pairs, key=lambda p: len(pair_candidates[p]))
    
    used_interior = set()
    assignment = {}
    
    def backtrack(idx):
        nonlocal used_interior
        if time.time() > deadline:
            return False
        if idx == len(pairs_ordered):
            return True
        u, v = pairs_ordered[idx]
        for path in pair_candidates[(u, v)]:
            interior = set(path[1:-1])
            if interior.isdisjoint(used_interior):
                assignment[(u, v)] = path
                used_interior.update(interior)
                if backtrack(idx + 1):
                    return True
                used_interior -= interior
                del assignment[(u, v)]
        return False
    
    if backtrack(0):
        return True, assignment
    return False, {}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    adj = _build_adj()
    
    # Step 1: Find maximum clique
    clique = _max_clique(adj, deadline)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    branches = sorted(clique)
    
    # Verify the clique has valid disjoint paths
    success, paths_dict = _backtrack_paths(adj, branches, deadline)
    if not success:
        # Fallback to just an edge
        u, v = P['edges'][0]
        branches = [u, v]
        success, paths_dict = _backtrack_paths(adj, branches, deadline)
    
    # Step 2: Try to grow the branch set by adding non-branch vertices
    branch_set = set(branches)
    candidates = [v for v in range(n) if v not in branch_set]
    
    # Sort candidates by degree (higher degree first)
    candidates.sort(key=lambda v: -len(adj[v]))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in candidates[:]:
            if cand in branch_set:
                continue
            if time.time() > deadline:
                break
            new_branches = branches + [cand]
            s, _ = _backtrack_paths(adj, new_branches, deadline)
            if s:
                branches.append(cand)
                branch_set.add(cand)
                candidates.remove(cand)
                improved = True
    
    # Final path extraction
    success, paths_dict = _backtrack_paths(adj, branches, deadline)
    if not success:
        # Fallback: use clique paths
        branches = sorted(clique)
        success, paths_dict = _backtrack_paths(adj, branches, deadline)
        if not success:
            u, v = P['edges'][0]
            branches = [u, v]
            paths_dict = {(u, v): [u, v]}
    
    # Build output
    paths = []
    for u, v in combinations(sorted(branches), 2):
        if (u, v) in paths_dict:
            paths.append(list(paths_dict[(u, v)]))
        elif (v, u) in paths_dict:
            paths.append(list(paths_dict[(v, u)]))
        else:
            paths.append([u, v])
    
    return {'branches': [int(b) for b in branches], 'paths': paths}

# EVOLVE-BLOCK-END
