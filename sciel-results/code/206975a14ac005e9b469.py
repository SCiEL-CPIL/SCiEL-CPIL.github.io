"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(coloring, edges, n):
    colors_used = set(coloring)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
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
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not union(u, v):
                        return False
    return True

def _is_valid(coloring, edges, n):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    if not _check_bicolored_forest(coloring, edges, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    edges = P['edges']
    
    # Precompute neighbors
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    # Try increasing number of colors
    for num_colors in range(2, 50):
        if time.time() > deadline:
            break
        
        # Try a few deterministic patterns
        for pattern_seed in range(10):
            if time.time() > deadline:
                break
            if pattern_seed == 0:
                coloring = [v % num_colors for v in range(n)]
            elif pattern_seed == 1:
                coloring = [(v * 3) % num_colors for v in range(n)]
            elif pattern_seed == 2:
                coloring = [(v * 7) % num_colors for v in range(n)]
            elif pattern_seed == 3:
                coloring = [(v * 11) % num_colors for v in range(n)]
            elif pattern_seed == 4:
                coloring = [(v * 13) % num_colors for v in range(n)]
            else:
                rng = random.Random(pattern_seed * 1000 + num_colors)
                coloring = [rng.randint(0, num_colors - 1) for _ in range(n)]
            
            if _is_valid(coloring, edges, n):
                return [int(c) for c in coloring]
        
        # Try random search
        for attempt in range(100):
            if time.time() > deadline:
                break
            coloring = [random.randint(0, num_colors - 1) for _ in range(n)]
            if _is_valid(coloring, edges, n):
                return [int(c) for c in coloring]
    
    # Fallback: all distinct colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
