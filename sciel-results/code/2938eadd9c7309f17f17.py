"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for fast iteration
    edge_list = edges
    
    def check_bicolored_cycles(coloring):
        """Check if any pair of colors induces a cycle using union-find."""
        # Group vertices by color
        color_to_vertices = defaultdict(list)
        for i, c in enumerate(coloring):
            color_to_vertices[c].append(i)
        
        colors = list(color_to_vertices.keys())
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                ci, cj = colors[i], colors[j]
                # Only check if both colors have at least one vertex
                if not color_to_vertices[ci] or not color_to_vertices[cj]:
                    continue
                
                # Union-find for vertices with color ci or cj
                parent = {}
                
                def find(x):
                    if x not in parent:
                        parent[x] = x
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return True  # Cycle found
                    parent[rx] = ry
                    return False
                
                # Only check edges between ci and cj
                for u, v in edge_list:
                    cu, cv = coloring[u], coloring[v]
                    if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                        if union(u, v):
                            return True  # Bicolored cycle found
        return False
    
    def is_valid(coloring):
        """Check proper coloring and no bicolored cycles."""
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        return not check_bicolored_cycles(coloring)
    
    def count_colors(coloring):
        return len(set(coloring))
    
    best = list(range(n))  # Fallback: 50 colors, always valid
    best_count = n
    
    # Phase 1: Try structured modular colorings (very fast)
    # color[i] = (a*i + b) % k
    # We try k=3 first, then 4, then 5.
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        for a in range(1, k):
            if time.time() > deadline:
                break
            for b in range(k):
                if time.time() > deadline:
                    break
                coloring = [(i * a + b) % k for i in range(n)]
                
                # Quick proper check
                proper = True
                for u, v in edge_list:
                    if coloring[u] == coloring[v]:
                        proper = False
                        break
                if not proper:
                    continue
                
                # Cycle check
                if not check_bicolored_cycles(coloring):
                    cnt = count_colors(coloring)
                    if cnt < best_count:
                        best = coloring[:]
                        best_count = cnt
                    if cnt == 3:
                        return [int(x) for x in best]
    
    # Phase 2: Backtracking with forward checking for k=3
    if best_count > 3 and time.time() < deadline:
        bt_deadline = time.time() + 20.0
        colors = [-1] * n
        
        # Order vertices by degree (all 4) or just 0..n-1
        order = list(range(n))
        
        def bt3(idx):
            if time.time() > bt_deadline:
                return False
            if idx == len(order):
                return not check_bicolored_cycles(colors)
            v = order[idx]
            for c in range(3):
                # Forward checking: ensure no neighbor has this color
                ok = True
                for u in adj[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                colors[v] = c
                if bt3(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        if bt3(0):
            best = colors[:]
            best_count = 3
            return [int(x) for x in best]
    
    # Phase 3: Backtracking for k=4 if we haven't found a valid 4-coloring
    if best_count > 4 and time.time() < deadline:
        bt_deadline = time.time() + 15.0
        colors = [-1] * n
        order = list(range(n))
        
        def bt4(idx):
            if time.time() > bt_deadline:
                return False
            if idx == len(order):
                return not check_bicolored_cycles(colors)
            v = order[idx]
            for c in range(4):
                ok = True
                for u in adj[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                colors[v] = c
                if bt4(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        if bt4(0):
            best = colors[:]
            best_count = 4
            return [int(x) for x in best]
    
    # Fallback: Ensure we return something valid
    if not is_valid(best):
        best = list(range(n))
        best_count = n
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
