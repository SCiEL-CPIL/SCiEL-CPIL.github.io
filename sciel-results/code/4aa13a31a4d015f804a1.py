"""Optimized backtracking for bicolored-forest coloring on the pinned degree-four graph."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

def _validate(coloring, edges):
    """Check: adjacent vertices differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in elist:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    
    n = N
    edges = EDGES
    
    # Vertex ordering: use degree-based heuristic (highest degree first)
    degrees = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try k=3 first
    for k in range(3, 8):
        if time.time() > deadline:
            break
        
        coloring = [-1] * n
        pair_uf = {}  # key -> (parent, rank)
        
        def try_assign(v, c):
            for nb in adj[v]:
                if coloring[nb] == c:
                    return False, [], {}
            affected_keys = []
            for nb in adj[v]:
                cn = coloring[nb]
                if cn != -1 and cn != c:
                    key = (min(cn, c), max(cn, c))
                    if key not in affected_keys:
                        affected_keys.append(key)
            snapshots = {}
            for key in affected_keys:
                if key in pair_uf:
                    snapshots[key] = (pair_uf[key][0][:], pair_uf[key][1][:])
            valid = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(cn, c), max(cn, c))
                if key not in pair_uf:
                    parent = list(range(n))
                    rank = [0] * n
                    pair_uf[key] = (parent, rank)
                parent, rank = pair_uf[key]
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                ru, rv = find(v), find(nb)
                if ru == rv:
                    valid = False
                    break
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
            if not valid:
                return False, [], {}
            return True, affected_keys, snapshots
        
        def undo_assign(v, c, affected_keys, snapshots):
            coloring[v] = -1
            for key in affected_keys:
                if key in snapshots:
                    pair_uf[key] = (snapshots[key][0], snapshots[key][1])
                elif key in pair_uf:
                    del pair_uf[key]
        
        def backtrack(pos):
            if time.time() > deadline:
                return None
            if pos == n:
                return coloring[:]
            v = order[pos]
            forbidden = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    forbidden.add(coloring[nb])
            for c in range(k):
                if c in forbidden:
                    continue
                ok, ak, sn = try_assign(v, c)
                if ok:
                    coloring[v] = c
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                    undo_assign(v, c, ak, sn)
            return None
        
        result = backtrack(0)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Fallback: greedy
    greedy = [0] * n
    for v in range(n):
        forbidden = set()
        for nb in adj[v]:
            if greedy[nb] != -1:
                forbidden.add(greedy[nb])
        c = 0
        while c in forbidden:
            c += 1
        greedy[v] = c
    if _validate(greedy, edges):
        return [int(c) for c in greedy]
    
    return list(range(n))
