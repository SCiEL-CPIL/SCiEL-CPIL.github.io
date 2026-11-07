"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def is_valid(coloring, k):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
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
                            return False
        return True
    
    def backtrack_search(k):
        coloring = [-1] * n
        
        def bt(v):
            if time.time() > deadline:
                return None
            if v == n:
                return coloring[:] if is_valid(coloring, k) else None
            
            for c in range(k):
                ok = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                coloring[v] = c
                result = bt(v + 1)
                if result is not None:
                    return result
                coloring[v] = -1
            
            return None
        
        return bt(0)
    
    for k in [3, 4, 5]:
        if time.time() > deadline - 5.0:
            break
        result = backtrack_search(k)
        if result is not None:
            return result
    
    # Fallback greedy
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return list(coloring)
# EVOLVE-BLOCK-END

