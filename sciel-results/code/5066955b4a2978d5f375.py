"""Rewritten program: search for a large clique subdivision using branch-and-bound + local search."""
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    from collections import deque
    from itertools import combinations

    deadline = time.time() + 50  # safety margin before 60s (or 90s) kill
    n = P['n']
    edges = P['edges']
    edge_set = set()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        adj[u].add(v)
        adj[v].add(u)

    # Precompute degree ordering: try high-degree vertices first
    deg = [len(adj[i]) for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: -deg[x])

    # BFS shortest path between two vertices avoiding a set of blocked vertices
    def bfs_path(s, t, blocked):
        """Find shortest path from s to t avoiding blocked (interior). s,t not in blocked."""
        if s == t:
            return [s]
        if s in blocked or t in blocked:
            return None
        # BFS
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                # reconstruct
                path = [u]
                while path[-1] != s:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            for v in adj[u]:
                if v not in parent and v not in blocked:
                    parent[v] = u
                    q.append(v)
        return None

    # Check if a set of branch vertices has a valid subdivision
    def check_subdivision(branches):
        """
        For a given set of branch vertices, try to find internally disjoint paths
        for every pair. Returns dict with 'branches' and 'paths' if valid, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        # blocked = all branch vertices (paths must avoid them in interior)
        blocked = set(branch_set)

        # We need to assign a path to each pair such that all interiors are disjoint.
        # Greedy: for each pair, find shortest path avoiding already-used interior vertices.
        used_interior = set()
        paths = []
        for i, j in pairs:
            s, t = branches[i], branches[j]
            # Find path avoiding used_interior and all branch vertices (except endpoints)
            # blocked for this path: all branch vertices + used_interior
            blk = blocked | used_interior
            path = bfs_path(s, t, blk)
            if path is None:
                return None
            # Extract interior
            interior = set(path[1:-1])
            if interior & branch_set:
                return None  # should not happen since branch vertices are blocked
            if interior & used_interior:
                return None
            used_interior |= interior
            paths.append(path)

        return {'branches': list(branches), 'paths': paths}

    # Branch and bound: try to build a large branch set
    best_result = None
    best_k = 0

    # Start with a greedy seed: pick high-degree vertices
    def greedy_seed():
        chosen = []
        for v in order:
            if len(chosen) >= n:
                break
            # Check if adding v keeps the set a clique in the "subdivision sense"
            # For now, just try to add and check later
            chosen.append(v)
        return chosen

    # Better: start with a small clique and expand
    # Find all cliques of size up to some limit
    def find_cliques(max_size=8):
        """Find cliques up to max_size using Bron-Kerbosch."""
        cliques = []
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            # Pick pivot
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]) + len(X & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                if len(R) + len(P) <= best_k and len(R) >= max_size:
                    continue
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        bronk(set(), set(range(n)), set())
        return cliques

    # First try: find a good clique and check if it has a subdivision
    # Since the seed program found a triangle (3 branches), let's search for larger sets.
    
    # Strategy: 
    # 1. Try all subsets of high-degree vertices of increasing size
    # 2. For each subset, check if a valid subdivision exists
    # 3. Use local search to improve

    # Get top-degree vertices
    top_k = min(n, 15)
    top_vertices = order[:top_k]

    # Try sizes from large to small
    found = False
    for k in range(min(top_k, 10), 1, -1):
        if time.time() > deadline:
            break
        # Try combinations of top vertices
        for combo in combinations(top_vertices, k):
            if time.time() > deadline:
                break
            res = check_subdivision(list(combo))
            if res is not None:
                if best_result is None or k > best_k:
                    best_result = res
                    best_k = k
                found = True
                # Don't break: try to find even better (but we're going large to small, so first found is best)
                # Actually we want the largest, so if we found one at this k, we can stop
                # But there might be a larger k we haven't tried... no, we're going from large to small
                # So the first found at the largest k is optimal among top_vertices
                # But maybe a set not entirely in top_vertices is larger?
                # Let's continue searching for now
        if found:
            # We found something at this k, but maybe a larger k exists with different vertices
            # Continue to check if we can do better
            pass

    # If we found a result, try to improve with local search
    if best_result is not None:
        # Local search: try swapping one branch for another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_branches = set(best_result['branches'])
            for v in range(n):
                if v in cur_branches:
                    continue
                if time.time() > deadline:
                    break
                for u in list(cur_branches):
                    new_set = (cur_branches - {u}) | {v}
                    res = check_subdivision(list(new_set))
                    if res is not None:
                        if len(res['branches']) > len(best_result['branches']):
                            best_result = res
                            best_k = len(res['branches'])
                            improved = True
                            break
                if improved:
                    break

    # If nothing found, fall back to a simple edge
    if best_result is None:
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}

    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result
