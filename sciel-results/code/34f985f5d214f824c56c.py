"""Rewritten program: minimize distinct colors with bicolored-cycle-free constraint."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(n, edges, coloring, c1, c2):
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
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u, v in edges:
        if coloring[u] == c1 and coloring[v] == c2:
            if union(u, v):
                return True
        elif coloring[u] == c2 and coloring[v] == c1:
            if union(u, v):
                return True
    return False

def _is_valid(n, edges, coloring, num_colors):
    """Check all constraints: proper coloring + no bicolored cycles."""
    # Proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Bicolored cycle check for each pair
    for c1 in range(num_colors):
        for c2 in range(c1 + 1, num_colors):
            if _has_bicolored_cycle(n, edges, coloring, c1, c2):
                return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    
    # Try modular colorings c_i = (a*i + b) % k
    # The graph is C(50, {1, 7}). 3 colors are sufficient.
    for k in range(3, 6):
        for a in range(1, k):
            for b in range(k):
                coloring = [(a * i + b) % k for i in range(n)]
                if _is_valid(n, edges, coloring, k):
                    return [int(c) for c in coloring]
    
    # Fallback: Greedy coloring
    coloring = [0] * n
    for v in range(n):
        neighbor_colors = set(coloring[nb] for nb in adj[v])
        c = 0
        while c in neighbor_colors:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
