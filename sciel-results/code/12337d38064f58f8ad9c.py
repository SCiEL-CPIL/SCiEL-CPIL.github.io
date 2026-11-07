"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    edge_list = [(u, v) for u, v in edges]
    
    def check_bicolor_forest(colors, max_color):
        """Check that for every pair of colors (a,b), induced subgraph is a forest."""
        # Only check pairs where both colors appear
        used_colors = set(c for c in colors if c != -1)
        color_list = sorted(used_colors)
        for i, a in enumerate(color_list):
            for b in color_list[i+1:]:
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for u, v in edge_list:
                    cu, cv = colors[u], colors[v]
                    if (cu == a and cv == b) or (cu == b and cv == a):
                        ru, rv = find(u), find(v)
                        if ru == rv:
                            return False
                        parent[ru] = rv
        return True
    
    # Vertex ordering: highest degree first
    degrees = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    def backtrack(colors, step, max_color):
        if time.time() > deadline:
            return None
        if step == n:
            return colors[:]
        
        v = order[step]
        for c in range(max_color + 1):
            # Check adjacency with already-colored neighbors
            valid = True
            for nb in adj[v]:
                if pos[nb] < step and colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            if check_bicolor_forest(colors, max_color):
                result = backtrack(colors, step + 1, max_color)
                if result is not None:
                    return result
            colors[v] = -1
        
        return None
    
    # Try with increasing number of colors
    for max_color in range(3, 7):
        colors = [-1] * n
        result = backtrack(colors, 0, max_color)
        if result is not None:
            return result
    
    # Fallback: return 50 distinct colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
