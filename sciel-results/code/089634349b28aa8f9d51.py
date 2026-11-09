"""Find large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_ADJ = None
_N = None

def _adjacency():
    global _ADJ, _N
    if _ADJ is None:
        _N = P['n']
        _ADJ = [set() for _ in range(_N)]
        for u, v in P['edges']:
            _ADJ[u].add(v)
            _ADJ[v].add(u)
    return _ADJ

def _bfs_path(src, dst, blocked, adj):
    """Find any simple path from src to dst avoiding blocked (interior). Returns list or None."""
    if src == dst:
        return None
    from collections import deque
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            # reconstruct
            path = []
            cur = dst
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for w in adj[u]:
            if w not in parent:
                if w == dst or w not in blocked:
                    parent[w] = u
                    q.append(w)
    return None

def _verify(branches, adj, n):
    """Given an ordered branch list, try to assign disjoint simple paths for every pair.
    Returns dict of paths (tuple pair -> path list) or None if infeasible."""
    k = len(branches)
    bset = set(branches)
    # Build list of pairs
    pairs = list(combinations(range(k), 2))
    # Precompute interior vertices (non-branch)
    # We'll assign paths greedily, choosing the pair that currently has the fewest options
    # and then try to find a path avoiding already-used interior vertices.
    assigned = {}  # (i,j) -> path
    used_interior = set()
    # Order pairs: try to assign pairs that share branch endpoints that have many other pairs first? 
    # Simpler: just iterate. But to be robust, try a few orderings.
    # For now, single deterministic order with backtracking-lite: if a pair fails, try swapping with a later unassigned pair.
    
    # Strategy: repeatedly pick the unassigned pair with the smallest number of available simple paths (bounded)
    # To keep it fast, we just do BFS and if multiple, pick shortest.
    
    # We'll do a simple sequential assignment with one level of retry.
    order = list(range(len(pairs)))
    # Heuristic: pairs sharing the same branch with high degree first
    # Compute degree within branch set? Just use global degree.
    def pair_key(idx):
        i, j = pairs[idx]
        return -(len(adj[branches[i]]) + len(adj[branches[j]]))
    order.sort(key=pair_key)
    
    def try_assign(order):
        assigned = {}
        used_interior = set()
        for idx in order:
            i, j = pairs[idx]
            s, t = branches[i], branches[j]
            # Find a path avoiding used_interior and all branches except s,t
            blocked = used_interior | (bset - {s, t})
            path = _bfs_path(s, t, blocked, adj)
            if path is None:
                return None
            # Mark interior as used
            for v in path[1:-1]:
                if v in bset:
                    return None  # should not happen
                used_interior.add(v)
            assigned[(i, j)] = path
        return assigned
    
    res = try_assign(order)
    if res is not None:
        return res
    # Try a couple of alternative orderings
    # Reverse
    res = try_assign(list(reversed(order)))
    if res is not None:
        return res
    # Random-ish: sort by smaller degree
    def pair_key2(idx):
        i, j = pairs[idx]
        return (len(adj[branches[i]]), len(adj[branches[j]]))
    order2 = sorted(range(len(pairs)), key=pair_key2)
    res = try_assign(order2)
    if res is not None:
        return res
    return None

def construct():
    adj = _adjacency()
    n = _N
    deadline = time.time() + 75.0  # safety margin before 90s kill
    best = None  # (size, branches, paths_dict)
    
    # Start with a maximal clique via greedy
    def greedy_clique():
        # Bronk-Kerbosch style but greedy: repeatedly add vertex with max edges to current clique
        clique = []
        remaining = list(range(n))
        # Start with highest degree vertex
        remaining.sort(key=lambda v: -len(adj[v]))
        clique.append(remaining[0])
        remaining.remove(remaining[0])
        changed = True
        while changed and remaining:
            changed = False
            best_v = None
            best_count = -1
            for v in remaining:
                c = sum(1 for u in clique if u in adj[v])
                if c == len(clique) and c > best_count:
                    best_count = c
                    best_v = v
                elif c > best_count:
                    best_count = c
                    best_v = v
            if best_v is not None:
                clique.append(best_v)
                remaining.remove(best_v)
                changed = True
        return clique
    
    # Try to grow beyond clique by adding non-clique vertices that still allow disjoint paths
    # We'll maintain a branch set and a valid path assignment.
    
    def try_grow(branches, paths_dict):
        """Try to add one more vertex. Returns (new_branches, new_paths) or None."""
        bset = set(branches)
        candidates = [v for v in range(n) if v not in bset]
        # Sort candidates by degree
        candidates.sort(key=lambda v: -len(adj[v]))
        for v in candidates:
            new_branches = branches + [v]
            new_paths = _verify(new_branches, adj, n)
            if new_paths is not None:
                return new_branches, new_paths
        return None
    
    # Initial: greedy clique
    clique = greedy_clique()
    # Verify clique
    paths = _verify(clique, adj, n)
    if paths is not None:
        best = (len(clique), clique, paths)
    
    # Now grow
    branches = clique
    paths_dict = paths
    # Iteratively try to add vertices
    while time.time() < deadline:
        grown = try_grow(branches, paths_dict)
        if grown is None:
            break
        branches, paths_dict = grown
        if len(branches) > best[0]:
            best = (len(branches), branches, paths_dict)
        # Continue growing
        # Also try different starting points? For now, keep going.
    
    # Also try starting from different maximal cliques (a few)
    # Generate a few random-ish maximal cliques by different starting vertices
    tried_starts = set()
    for start in range(n):
        if time.time() > deadline:
            break
        # Greedy clique starting from 'start'
        clique = [start]
        remaining = [v for v in range(n) if v != start]
        # Add vertices that connect to all in clique, prefer high degree
        changed = True
        while changed:
            changed = False
            best_v = None
            best_deg = -1
            for v in remaining:
                if all(v in adj[u] for u in clique):
                    d = len(adj[v])
                    if d > best_deg:
                        best_deg = d
                        best_v = v
            if best_v is not None:
                clique.append(best_v)
                remaining.remove(best_v)
                changed = True
        # Verify
        paths = _verify(clique, adj, n)
        if paths is not None:
            if len(clique) > best[0]:
                best = (len(clique), clique, paths)
            # Try to grow from this clique
            branches = clique
            paths_dict = paths
            while time.time() < deadline:
                grown = try_grow(branches, paths_dict)
                if grown is None:
                    break
                branches, paths_dict = grown
                if len(branches) > best[0]:
                    best = (len(branches), branches, paths_dict)
    
    # Fallback: at least 2 branches (an edge)
    if best is None:
        u, v = P['edges'][0]
        best = (2, [u, v], {(0, 1): [u, v]})
    
    size, branches, paths_dict = best
    # Build output paths for every unordered pair
    out_paths = []
    for i in range(size):
        for j in range(i + 1, size):
            out_paths.append(list(paths_dict[(i, j)]))
    
    return {'branches': list(branches), 'paths': out_paths}
# EVOLVE-BLOCK-END
