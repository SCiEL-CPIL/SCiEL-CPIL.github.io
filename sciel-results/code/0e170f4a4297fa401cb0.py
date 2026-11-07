"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def has_bicolored_cycle(coloring, max_color):
        """Check if any pair of colors forms a cycle."""
        for c1 in range(max_color):
            for c2 in range(c1 + 1, max_color):
                # Union-find on subgraph induced by colors c1, c2
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
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                for u, v in edges:
                    cu, cv = coloring[u], coloring[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if not union(u, v):
                            return True
        return False
    
    def is_valid(coloring, max_color):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        return not has_bicolored_cycle(coloring, max_color)
    
    # Backtracking with MRV
    def solve(max_colors):
        coloring = [-1] * n
        domains = [set(range(max_colors)) for _ in range(n)]
        
        def get_mrv_var():
            """Find uncolored vertex with fewest available colors."""
            best = -1
            best_count = max_colors + 1
            for v in range(n):
                if coloring[v] == -1:
                    count = len(domains[v])
                    if count < best_count:
                        best_count = count
                        best = v
                    if count == 0:
                        return v
            return best
        
        def backtrack():
            v = get_mrv_var()
            if v == -1:
                return True  # All colored
            if v >= 0 and len(domains[v]) == 0:
                return False
            
            for c in sorted(domains[v]):
                coloring[v] = c
                saved = {}
                valid = True
                
                for nb in adj[v]:
                    if coloring[nb] == -1:
                        if c in domains[nb]:
                            saved[nb] = domains[nb].copy()
                            domains[nb].discard(c)
                            if not domains[nb]:
                                valid = False
                                break
                    else:
                        if coloring[nb] == c:
                            valid = False
                            break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                for nb, old in saved.items():
                    domains[nb] = old
                coloring[v] = -1
            
            return False
        
        return backtrack()
    
    deadline = time.time() + 50
    
    for mc in [3, 4, 5]:
        if time.time() >= deadline:
            break
        if solve(mc):
            if is_valid(coloring, mc):
                return [int(c) for c in coloring]
    
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
