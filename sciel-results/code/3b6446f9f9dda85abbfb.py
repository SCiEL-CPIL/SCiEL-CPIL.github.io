"""Rewritten program: bicolored-forest coloring with optimized backtracking and constraint propagation."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 85
    
    # Try increasing number of colors
    for k in range(2, 15):
        if time.time() > deadline - 5:
            break
        res = _solve_k(n, adj, k, deadline)
        if res is not None:
            return res
    
    # Fallback: greedy coloring
    res = _greedy(n, adj)
    if res is not None:
        return res
    return [int(i) for i in range(n)]


def _greedy(n, adj):
    color = [-1] * n
    # Order by degree descending for better greedy
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    for v in order:
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(50):
            if c not in used:
                color[v] = c
                break
    if -1 in color:
        return None
    return [int(c) for c in color]


def _solve_k(n, adj, k, deadline):
    """Solve with exactly k colors using backtracking with constraint propagation."""
    
    # Build constraint structure: for each color pair, track edges
    # We'll use a smarter approach: try to find a valid assignment
    
    # Use a heuristic ordering: vertices with higher degree first
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    # Reverse order for backtracking (assign high-degree first)
    backtrack_order = order
    
    color = [-1] * n
    # For each pair of colors, track the union-find structure
    # We'll rebuild when needed
    
    # Precompute: for each vertex, which other vertices are adjacent
    # and their indices
    
    # Use a recursive backtracking with forward checking
    best_result = [None]
    
    def find_cycle_in_subgraph(subgraph_edges, vertices_in_subgraph):
        """Check if subgraph induced by vertices with colors in a pair has a cycle."""
        if not subgraph_edges:
            return False
        # Build adjacency for the subgraph
        adj_sub = defaultdict(list)
        for u, v in subgraph_edges:
            adj_sub[u].append(v)
            adj_sub[v].append(u)
        # DFS to detect cycle
        visited = set()
        def dfs(node, parent):
            visited.add(node)
            for nei in adj_sub[node]:
                if nei == parent:
                    continue
                if nei in visited:
                    return True
                if dfs(nei, node):
                    return True
            return False
        for v in vertices_in_subgraph:
            if v not in visited:
                if dfs(v, -1):
                    return True
        return False
    
    def check_bicolored_forest(c1, c2):
        """Check if the subgraph induced by vertices with color c1 or c2 is a forest."""
        # Get all edges where both endpoints have color c1 or c2
        edges_in_pair = []
        vertices_in_pair = []
        for v in range(n):
            if color[v] == c1 or color[v] == c2:
                vertices_in_pair.append(v)
        vset = set(vertices_in_pair)
        for u, v in P['edges']:
            if color[u] in (c1, c2) and color[v] in (c1, c2):
                edges_in_pair.append((u, v))
        # A forest with V vertices has at most V-1 edges per connected component
        # Total edges in a forest <= V - (number of components)
        # Simple check: edges <= vertices (necessary but not sufficient)
        # Better: use union-find
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in edges_in_pair:
            if not union(u, v):
                return False  # Cycle detected
        return True
    
    def backtrack(idx):
        if time.time() > deadline - 3:
            return None
        if idx == n:
            return list(color)
        v = backtrack_order[idx]
        
        # Try each color
        for c in range(k):
            # Check adjacency constraint
            valid = True
            for u in adj[v]:
                if color[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Tentatively assign
            color[v] = c
            
            # Check all color pairs involving c
            ok = True
            for cu in range(k):
                if cu == c:
                    continue
                # Only check if there are vertices with color cu
                has_cu = any(color[x] == cu for x in range(n) if x != v)
                if not has_cu:
                    continue
                if not check_bicolored_forest(c, cu):
                    ok = False
                    break
            
            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            color[v] = -1
        
        return None
    
    return backtrack(0)
