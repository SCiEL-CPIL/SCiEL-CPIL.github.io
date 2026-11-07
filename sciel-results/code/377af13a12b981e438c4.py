import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def is_valid(coloring):
    n = P['n']
    edges = P['edges']
    
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
            
    # Check forest condition for all pairs of colors
    colors_used = set(coloring)
    
    if len(colors_used) < 2:
        return True
        
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = [i for i in range(n) if coloring[i] in (c1, c2)]
            v_set = set(verts)
            
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
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
            
            for u, v in edges:
                if u in v_set and v in v_set:
                    if not union(u, v):
                        return False
                        
    return True

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with safety margin
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find colorings with increasing number of colors
    # Use backtracking with constraint propagation
    
    def backtrack(v, k, coloring, adj):
        """Try to color vertex v with one of colors 0..k-1"""
        if time.time() > deadline:
            return None
        
        if v == n:
            return coloring[:]
        
        # Get available colors (not used by neighbors)
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                neighbor_colors.add(coloring[nb])
        
        for c in range(k):
            if c not in neighbor_colors:
                coloring[v] = c
                result = backtrack(v + 1, k, coloring, adj)
                if result is not None:
                    return result
                coloring[v] = -1
        
        return None
    
    # Try small numbers of colors first
    best_coloring = None
    
    for k in range(2, 8):  # Try 2 to 7 colors
        if time.time() > deadline:
            break
        
        coloring = [-1] * n
        result = backtrack(0, k, coloring, adj)
        
        if result is not None:
            # Check if this coloring satisfies the forest condition
            if is_valid(result):
                best_coloring = result
                break
    
    if best_coloring is not None:
        return [int(c) for c in best_coloring]
    
    # Fallback: try random assignments with more colors
    import random
    for k in range(8, 12):
        if time.time() > deadline:
            break
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            coloring = [random.randint(0, k-1) for _ in range(n)]
            if is_valid(coloring):
                return [int(c) for c in coloring]
    
    # Final fallback: unique colors (always valid)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
