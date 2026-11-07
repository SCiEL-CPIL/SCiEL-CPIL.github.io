"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    adj = [set() for _ in range(P['n'])]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest_fast(coloring, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    colors_used = set(coloring)
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            parent = list(range(n))
            has_cycle = False
            
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def _count_distinct(coloring):
    return len(set(coloring))


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Start with a valid coloring: use n colors (one per vertex)
    best_coloring = list(range(n))
    best_count = n
    
    # Try to find valid colorings with increasing k
    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random seeds and strategies
        found = False
        for seed in range(200):
            if time.time() > deadline - 3.0:
                break
            rng = random.Random(seed + k * 1000)
            
            # Strategy 1: Pattern-based coloring
            # Try color = (a*i + b) % k for various a, b
            for a in range(1, k + 1):
                for b in range(k):
                    coloring = [((a * i + b) % k) for i in range(n)]
                    valid = True
                    for u, v in edges:
                        if coloring[u] == coloring[v]:
                            valid = False
                            break
                    if not valid:
                        continue
                    if _check_bicolored_forest_fast(coloring, adj, n):
                        best_coloring = coloring
                        best_count = _count_distinct(coloring)
                        found = True
                        break
                if found:
                    break
            if found:
                break
            
            # Strategy 2: Randomized greedy with backtracking-lite
            coloring = [-1] * n
            order = list(range(n))
            rng.shuffle(order)
            
            # Assign in random order
            success = True
            for idx in order:
                neighbor_colors = set()
                for nb in adj[idx]:
                    if coloring[nb] != -1:
                        neighbor_colors.add(coloring[nb])
                
                # Try colors in random order
                colors_to_try = list(range(k))
                rng.shuffle(colors_to_try)
                placed = False
                for c in colors_to_try:
                    if c not in neighbor_colors:
                        coloring[idx] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            
            if not success:
                continue
            
            if _check_bicolored_forest_fast(coloring, adj, n):
                best_coloring = coloring
                best_count = _count_distinct(coloring)
                found = True
                break
        
        if found:
            break
    
    # Ensure we return a list of Python ints
    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END
