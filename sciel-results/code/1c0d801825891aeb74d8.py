"""Efficient search for minimum-color valid coloring with bicolored-forest constraint."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: proper + no bicolored cycles."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _try_affine_3color(adj, n, edges):
    """Try all affine 3-coloring patterns on the 5x10 grid."""
    # Vertex i: row = i // 10, col = i % 10
    for a in range(3):
        for b in range(3):
            for c0 in range(3):
                if a == 0 and b == 0:
                    continue  # constant, won't be proper
                coloring = [(a * (i // 10) + b * (i % 10) + c0) % 3 for i in range(n)]
                if _check_coloring(coloring, edges):
                    return coloring
    return None

def _backtrack_3color(adj, n, edges, deadline):
    """Backtracking for 3-coloring with incremental bicolored cycle check."""
    # Order vertices: try a structured order for grid
    order = list(range(n))
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i
    coloring = [-1] * n
    # For each color pair, track union-find state incrementally
    # But for simplicity, do BFS check on assignment
    
    def would_create_cycle(v, c):
        """Check if assigning color c to v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1:
                continue
            if cn == c:
                return False  # adjacent same color, invalid
            # BFS in subgraph induced by colors {c, cn}
            visited = set()
            stack = [v]
            visited.add(v)
            found_path = False
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w == nb and cur == v:
                        continue  # skip the direct edge we're checking
                    if w == v and cur == nb:
                        continue
                    cw = coloring[w]
                    if cw != c and cw != cn:
                        continue
                    if w == nb:
                        return True  # cycle found
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
        return False
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = order[pos]
        for c in range(3):
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            coloring[v] = c
            if would_create_cycle(v, c):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def _greedy_then_refine(adj, n, edges, deadline):
    """Greedy coloring then try to reduce via merging."""
    # Greedy with DSATUR-like heuristic
    coloring = [-1] * n
    uncolored = set(range(n))
    
    while uncolored:
        # Pick vertex with most colored neighbors (tie-break: degree)
        best_v = None
        best_score = -1
        for v in uncolored:
            colored_neighbors = sum(1 for nb in adj[v] if coloring[nb] != -1)
            score = colored_neighbors * 100 + len(adj[v])
            if score > best_score:
                best_score = score
                best_v = v
        # Find smallest available color
        used = set(coloring[nb] for nb in adj[best_v] if coloring[nb] != -1)
        for c in range(50):
            if c not in used:
                coloring[best_v] = c
                break
        uncolored.remove(best_v)
    
    # Try to reduce number of colors
    for target_k in range(3, 50):
        if time.time() > deadline - 2:
            break
        # Map to 0..target_k-1
        distinct = sorted(set(coloring))
        if len(distinct) <= target_k:
            if _check_coloring(coloring, edges):
                return [int(c) for c in coloring]
        color_map = {c: i % target_k for i, c in enumerate(distinct)}
        new_coloring = [color_map[c] for c in coloring]
        if _check_coloring(new_coloring, edges):
            return [int(c) for c in new_coloring]
    
    return [int(c) for c in coloring]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # 1. Try fast affine 3-coloring patterns
    result = _try_affine_3color(adj, n, edges)
    if result is not None:
        if _check_coloring(result, edges):
            return [int(c) for c in result]
    
    # 2. Backtracking for 3 colors
    if time.time() < deadline - 20:
        result = _backtrack_3color(adj, n, edges, deadline - 20)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # 3. Greedy + refine
    result = _greedy_then_refine(adj, n, edges, deadline)
    if result is not None and _check_coloring(result, edges):
        return [int(c) for c in result]
    
    # Fallback: return identity (50 colors, always valid)
    return [int(i) for i in range(n)]
