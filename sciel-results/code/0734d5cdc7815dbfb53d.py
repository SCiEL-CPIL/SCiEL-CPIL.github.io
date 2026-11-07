import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check(coloring):
    adj = _build_adj()
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))

    for (a, b), edges in pair_edges.items():
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        vlist = list(verts)
        idx = {v: i for i, v in enumerate(vlist)}
        par = list(range(len(vlist)))
        rn = [0] * len(vlist)

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def un(x, y):
            rx, ry = f(x), f(y)
            if rx == ry:
                return False
            if rn[rx] < rn[ry]:
                rx, ry = ry, rx
            par[ry] = rx
            if rn[rx] == rn[ry]:
                rn[rx] += 1
            return True

        for u, v in edges:
            if not un(idx[u], idx[v]):
                return False
    return True

def _backtrack(adj, nc, deadline, ordering, node_limit=50000):
    n = P['n']
    coloring = [-1] * n
    nodes_visited = 0

    def dfs(idx):
        nonlocal nodes_visited
        if time.time() > deadline:
            return None
        if nodes_visited > node_limit:
            return None
        
        nodes_visited += 1
        
        if idx == n:
            return coloring[:]
        
        v = ordering[idx]
        
        used_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_colors.add(coloring[nb])
        
        for c in range(nc):
            if c in used_colors:
                continue
            
            ok = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                
                # Check if v and nb are connected in {c, cn} subgraph
                # BFS from v to nb, avoiding direct edge
                found = False
                queue = [(v, 0)]
                visited = {v}
                while queue:
                    cur, dist = queue.pop(0)
                    for w in adj[cur]:
                        if w in visited:
                            continue
                        cw = coloring[w]
                        if cw == c or cw == cn:
                            if w == nb:
                                if dist > 0:
                                    found = True
                                    break
                            else:
                                visited.add(w)
                                queue.append((w, dist + 1))
                    if found:
                        break
                if found:
                    ok = False
                    break
            
            if ok:
                coloring[v] = c
                res = dfs(idx + 1)
                if res is not None:
                    return res
                coloring[v] = -1
                
        return None

    return dfs(0)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = _build_adj()
    
    # Snake ordering
    ordering = []
    for i in range(7):
        row = list(range(i*7, (i+1)*7))
        if i % 2 == 1:
            row.reverse()
        ordering.extend(row)
    
    for nc in [3, 4]:
        if time.time() > deadline:
            break
        result = _backtrack(adj, nc, deadline, ordering)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]
            
    # Fallback
    return [int(i) for i in range(P['n'])]
