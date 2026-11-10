"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    edge_list = [(u, v) for u, v in edges]
    
    def full_check(coloring):
        from collections import defaultdict
        color_edges = defaultdict(list)
        for u, v in edge_list:
            cu, cv = coloring[u], coloring[v]
            if cu == cv:
                return False
            pair = (min(cu, cv), max(cu, cv))
            color_edges[pair].append((u, v))
        for pair, elist in color_edges.items():
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
            for u, v in elist:
                if not union(u, v):
                    return False
        return True
    
    # Try structured patterns first
    for k in range(2, 7):
        for offset in range(k):
            coloring = [((v // 10) + 2 * (v % 10) + offset) % k for v in range(n)]
            if full_check(coloring):
                return [int(c) for c in coloring]
    
    for k in range(2, 7):
        for offset in range(k):
            coloring = [(2 * (v // 10) + (v % 10) + offset) % k for v in range(n)]
            if full_check(coloring):
                return [int(c) for c in coloring]
    
    for k in range(2, 7):
        for offset in range(k):
            coloring = [((v // 10) + (v % 10) + offset) % k for v in range(n)]
            if full_check(coloring):
                return [int(c) for c in coloring]
    
    # Backtracking with incremental bicolored cycle checking
    degree_order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    def solve_with_cycles(max_colors):
        coloring = [-1] * n
        
        def bt(idx):
            if idx == n:
                return True
            v = degree_order[idx]
            for c in range(max_colors):
                ok = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                coloring[v] = c
                
                cycle_found = False
                for nb in adj[v]:
                    if coloring[nb] >= 0 and coloring[nb] != c:
                        cn = coloring[nb]
                        visited = set()
                        stack = [nb]
                        visited.add(nb)
                        while stack:
                            cur = stack.pop()
                            if cur == v:
                                cycle_found = True
                                break
                            for nb2 in adj[cur]:
                                if nb2 in visited:
                                    continue
                                cc = coloring[nb2]
                                if cc >= 0 and (cc == c or cc == cn):
                                    visited.add(nb2)
                                    stack.append(nb2)
                        if cycle_found:
                            break
                
                if cycle_found:
                    coloring[v] = -1
                    continue
                
                if bt(idx + 1):
                    return True
                coloring[v] = -1
            return False
        
        return bt(0)
    
    for mc in range(2, 8):
        if solve_with_cycles(mc):
            if full_check(coloring):
                return [int(c) for c in coloring]
    
    # Final fallback
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    if full_check(coloring):
        used_colors = sorted(set(coloring))
        remap = {c: i for i, c in enumerate(used_colors)}
        return [int(remap[c]) for c in coloring]
    
    return list(range(n))
# EVOLVE-BLOCK-END
