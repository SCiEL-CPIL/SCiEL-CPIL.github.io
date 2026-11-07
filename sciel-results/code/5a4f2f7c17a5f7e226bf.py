"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Helper to check if a coloring is valid
    def is_valid(coloring):
        # 1. Proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # 2. Bicolored forest constraint
        # For each pair of colors, check if induced subgraph is a forest
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2: continue
                # DSU for this pair
                parent = list(range(n))
                rank = [0] * n
                
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
                
                # Add edges between vertices of c1 and c2
                for u, v in edges:
                    if (coloring[u] == c1 and coloring[v] == c2) or (coloring[u] == c2 and coloring[v] == c1):
                        if not union(u, v):
                            return False
        return True

    # Try 4 colors first
    # Pattern: i % 4
    coloring = [i % 4 for i in range(n)]
    if is_valid(coloring):
        return [int(c) for c in coloring]
        
    # If simple pattern fails, use local search
    # We'll try to minimize the number of bicolored cycles
    def count_cycles(coloring):
        count = 0
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2: continue
                parent = list(range(n))
                rank = [0] * n
                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                def union(x, y):
                    nonlocal count
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        count += 1
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                for u, v in edges:
                    if (coloring[u] == c1 and coloring[v] == c2) or (coloring[u] == c2 and coloring[v] == c1):
                        union(u, v)
        return count

    best_coloring = coloring[:]
    best_cycles = count_cycles(coloring)
    
    # Simple hill climbing
    for _ in range(1000):
        if best_cycles == 0:
            break
        # Pick a random vertex
        v = random.randint(0, n-1)
        old_c = best_coloring[v]
        # Try a new color
        new_c = random.randint(0, 3)
        if new_c == old_c: continue
        
        # Check proper coloring
        valid = True
        for u in adj[v]:
            if best_coloring[u] == new_c:
                valid = False
                break
        if not valid: continue
        
        best_coloring[v] = new_c
        new_cycles = count_cycles(best_coloring)
        if new_cycles < best_cycles:
            best_cycles = new_cycles
        else:
            best_coloring[v] = old_c
            
    if is_valid(best_coloring):
        return [int(c) for c in best_coloring]

    # Try 5 colors
    coloring = [i % 5 for i in range(n)]
    if is_valid(coloring):
        return [int(c) for c in coloring]
        
    # Fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
