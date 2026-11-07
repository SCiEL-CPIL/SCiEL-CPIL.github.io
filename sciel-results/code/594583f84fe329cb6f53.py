"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj, n):
    remaining = set(range(n))
    order = []
    deg = {v: len(adj[v]) for v in range(n)}
    while remaining:
        min_deg = min(deg[v] for v in remaining)
        v = min(remaining, key=lambda x: (deg[x], x))
        order.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    return order

def _validate(coloring, adj, n):
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = sorted(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i+1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            parent = {}
            for v in range(n):
                if coloring[v] in (c1, c2):
                    parent[v] = v
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in P['edges']:
                if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        return False
                    parent[ru] = rv
    return True

def _solve(n, adj, max_colors, deadline):
    order = _degeneracy_order(adj, n)
    coloring = [-1] * n
    assigned = [False] * n
    node_count = [0]
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        node_count[0] += 1
        if idx == n:
            return list(coloring)
        
        v = order[idx]
        
        for c in range(max_colors):
            valid = True
            for u in adj[v]:
                if assigned[u] and coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            assigned[v] = True
            
            # Check bicolored forest constraint incrementally
            ok = True
            for u in adj[v]:
                if not assigned[u]:
                    continue
                cu = coloring[u]
                if cu == c:
                    continue
                # Check if adding edge (v,u) to subgraph of colors {c,cu} creates cycle
                c1, c2 = min(c, cu), max(c, cu)
                # Build DSU for this pair on assigned vertices
                parent = {}
                for w in range(n):
                    if assigned[w] and coloring[w] in (c1, c2):
                        parent[w] = w
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for a, b in P['edges']:
                    if assigned[a] and assigned[b] and coloring[a] in (c1, c2) and coloring[b] in (c1, c2):
                        ra, rb = find(a), find(b)
                        if ra == rb:
                            ok = False
                            break
                        parent[ra] = rb
                if not ok:
                    break
            
            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
            assigned[v] = False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 80
    adj = _build_adj()
    n = P['n']
    
    for max_colors in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]:
        if time.time() > deadline:
            break
        result = _solve(n, adj, max_colors, deadline)
        if result is not None:
            if _validate(result, adj, n):
                return [int(x) for x in result]
    
    # Fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
