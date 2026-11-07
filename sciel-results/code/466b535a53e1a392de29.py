"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

adj = defaultdict(list)
for e in EDGES:
    u, v = e
    adj[u].append(v)
    adj[v].append(u)

def check_validity(coloring):
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    color_pairs = defaultdict(list)
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            color_pairs[pair].append((u, v))
    for pair, edges in color_pairs.items():
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True
        for u, v in edges:
            if not union(u, v):
                return False
    return True

def solve_k(k, time_limit=80):
    """Try to find a valid k-coloring using backtracking with good ordering."""
    deadline = time.time() + time_limit
    coloring = [-1] * N
    
    # Vertex ordering: most constrained first (highest degree, then most colored neighbors)
    # Use static ordering by degree
    vertex_order = sorted(range(N), key=lambda v: -len(adj[v]))
    
    # For faster checking, precompute for each vertex its neighbors
    # Use forward checking
    import sys
    sys.setrecursionlimit(10000)
    
    result = [None]
    
    def backtrack(idx):
        if result[0] is not None:
            return
        if time.time() > deadline:
            return
        if idx == N:
            if check_validity(coloring):
                result[0] = coloring[:]
            return
        
        v = vertex_order[idx]
        # Get available colors
        used = set()
        for nb in adj[v]:
            if coloring[nb] >= 0:
                used.add(coloring[nb])
        
        # Try colors in order
        for c in range(k):
            if c not in used:
                coloring[v] = c
                backtrack(idx + 1)
                if result[0] is not None:
                    return
                coloring[v] = -1
    
    backtrack(0)
    return result[0]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    for k in range(2, 50):
        if time.time() > deadline:
            break
        remaining = max(1, deadline - time.time())
        result = solve_k(k, time_limit=min(remaining - 1, 60))
        if result is not None and check_validity(result):
            color_map = {}
            new_coloring = []
            for c in result:
                if c not in color_map:
                    color_map[c] = len(color_map)
                new_coloring.append(color_map[c])
            return [int(c) for c in new_coloring]
    
    # Fallback
    coloring = list(range(N))
    return coloring
# EVOLVE-BLOCK-END
