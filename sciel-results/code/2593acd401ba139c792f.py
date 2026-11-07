"""Rewritten program: minimize distinct colors with bicolored-cycle-free constraint."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _vertex_order(n, adj):
    """Order vertices by descending degree for better constraint propagation."""
    return sorted(range(n), key=lambda v: -len(adj[v]))

def _check_bicolored_pair(n, edges, coloring, c1, c2):
    """Check if subgraph induced by colors c1 and c2 has a cycle using union-find."""
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
            return True
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u, v in edges:
        cu = coloring[u]
        cv = coloring[v]
        if cu == -1 or cv == -1:
            continue
        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
            if union(u, v):
                return True
    return False

def _backtrack_search(n, edges, adj, k, deadline, vertex_order):
    """Backtracking with forward checking to find valid k-coloring."""
    coloring = [-1] * n
    
    def _check_new_assignment(coloring, v, c):
        """Check if assigning color c to vertex v creates any bicolored cycle."""
        for d in range(k):
            if d == c:
                continue
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
                    return True
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return False
            
            for u, w in edges:
                cu = coloring[u]
                cw = coloring[w]
                if cu in (c, d) and cw in (c, d) and cu != cw:
                    if union(u, w):
                        return False
        return True
    
    def _search(idx, depth):
        if time.time() > deadline:
            return None
        
        if idx == n:
            return list(coloring)
        
        v = vertex_order[idx]
        
        neighbor_colors = [coloring[nb] for nb in adj[v] if coloring[nb] != -1]
        color_counts = [0] * k
        for nc in neighbor_colors:
            if 0 <= nc < k:
                color_counts[nc] += 1
        
        color_order = sorted(range(k), key=lambda c: color_counts[c])
        
        for c in color_order:
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            coloring[v] = c
            if _check_new_assignment(coloring, v, c):
                result = _search(idx + 1, depth + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        
        return None
    
    return _search(0, 0)

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    vorder = _vertex_order(n, adj)
    
    deadline = time.time() + 85
    
    result = _backtrack_search(n, edges, adj, 3, deadline, vorder)
    if result is not None:
        return [int(c) for c in result]
    
    if time.time() < deadline:
        result = _backtrack_search(n, edges, adj, 4, deadline, vorder)
        if result is not None:
            return [int(c) for c in result]
    
    if time.time() < deadline:
        result = _backtrack_search(n, edges, adj, 5, deadline, vorder)
        if result is not None:
            return [int(c) for c in result]
    
    coloring = [0] * n
    for v in vorder:
        neighbor_colors = set(coloring[nb] for nb in adj[v])
        c = 0
        while c in neighbor_colors:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
