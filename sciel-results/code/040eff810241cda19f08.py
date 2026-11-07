"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _is_valid(colors, n, edges):
    """Check proper coloring and bicolored forest constraint."""
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Group edges by color pair
    color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        color_pairs[key].append((u, v))
    
    # For each color pair, check if induced subgraph is a forest
    for (c1, c2), pair_edges in color_pairs.items():
        verts = set()
        for u, v in pair_edges:
            verts.add(u)
            verts.add(v)
        
        # Union-Find
        parent = {v: v for v in verts}
        rank = {v: 0 for v in verts}
        
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
        
        for u, v in pair_edges:
            if not union(u, v):
                return False
    
    return True

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best = None
    best_count = n  # Start with 50 colors
    
    # Try random local search with decreasing color budgets
    for max_colors in range(4, 15):
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with random colors in [0, max_colors)
            colors = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Local search: try to fix violations
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Check validity
                if _is_valid(colors, n, edges):
                    distinct = len(set(colors))
                    if distinct < best_count:
                        best = list(colors)
                        best_count = distinct
                    break
                
                # Find a violating vertex and try to change its color
                # First, find adjacent same-color pairs
                violated = False
                for u, v in edges:
                    if colors[u] == colors[v]:
                        violated = True
                        break
                
                if not violated:
                    # No proper coloring violation, but maybe bicolored cycle
                    # Try random changes
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        v = random.randint(0, n - 1)
                        old_c = colors[v]
                        new_c = random.randint(0, max_colors - 1)
                        if new_c != old_c:
                            colors[v] = new_c
                            if _is_valid(colors, n, edges):
                                distinct = len(set(colors))
                                if distinct < best_count:
                                    best = list(colors)
                                    best_count = distinct
                                improved = True
                                break
                    continue
                
                # Change a random violating vertex
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    u, v = random.choice(edges)
                    if colors[u] == colors[v]:
                        target = random.choice([u, v])
                        old_c = colors[target]
                        # Try a new color
                        new_c = random.randint(0, max_colors - 1)
                        if new_c != old_c:
                            colors[target] = new_c
                            if _is_valid(colors, n, edges):
                                distinct = len(set(colors))
                                if distinct < best_count:
                                    best = list(colors)
                                    best_count = distinct
                                improved = True
                                break
                            # Even if not fully valid, keep the change if it reduces violations
                            # Count violations
                            colors[target] = new_c
                            break
            # End while improved
        
        # End for attempt
        if best is not None and best_count <= max_colors:
            # We found a solution with max_colors or fewer, try fewer
            continue
    
    if best is not None:
        return [int(c) for c in best]
    
    # Fallback
    return list(range(n))
# EVOLVE-BLOCK-END
