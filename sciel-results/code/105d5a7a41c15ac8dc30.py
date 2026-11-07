import time

P = {'n': 50, 'edges': [[0,1],[0,7],[0,43],[0,49],[1,2],[1,8],[1,44],[2,3],[2,9],[2,45],[3,4],[3,10],[3,46],[4,5],[4,11],[4,47],[5,6],[5,12],[5,48],[6,7],[6,13],[6,49],[7,8],[7,14],[8,9],[8,15],[9,10],[9,16],[10,11],[10,17],[11,12],[11,18],[12,13],[12,19],[13,14],[13,20],[14,15],[14,21],[15,16],[15,22],[16,17],[16,23],[17,18],[17,24],[18,19],[18,25],[19,20],[19,26],[20,21],[20,27],[21,22],[21,28],[22,23],[22,29],[23,24],[23,30],[24,25],[24,31],[25,26],[25,32],[26,27],[26,33],[27,28],[27,34],[28,29],[28,35],[29,30],[29,36],[30,31],[30,37],[31,32],[31,38],[32,33],[32,39],[33,34],[33,40],[34,35],[34,41],[35,36],[35,42],[36,37],[36,43],[37,38],[37,44],[38,39],[38,45],[39,40],[39,46],[40,41],[40,47],[41,42],[41,48],[42,43],[42,49],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49]]}

def construct():
    start = time.time()
    deadline = start + 85
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    for k in range(3, 50):
        if time.time() > deadline:
            break
        res = _solve(n, adj, k, deadline)
        if res is not None:
            return res
    return list(range(n))

def _solve(n, adj, k, deadline):
    colors = [-1] * n
    # DSU for each color pair
    # We'll use a dict: key = (min(c1,c2), max(c1,c2)), value = DSU
    # For rollback, we'll keep a stack of (pair_key, parent_change)
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For efficiency, precompute sorted neighbor lists
    adj_sorted = [sorted(adj[i]) for i in range(n)]
    
    def dsu_find(dsu, x):
        while dsu[x] != x:
            x = dsu[x]
        return x
    
    def dsu_union(dsu, x, y):
        rx, ry = dsu_find(dsu, x), dsu_find(dsu, y)
        if rx == ry:
            return None
        # Union by rank (simplified: just attach ry to rx)
        dsu[ry] = rx
        return (ry, rx)  # record for rollback
    
    def dsu_rollback(dsu, record):
        if record is not None:
            ry, rx = record
            dsu[ry] = ry
    
    dsu_store = {}  # (c1, c2) -> list (parent array)
    
    def get_dsu(c1, c2):
        key = (min(c1,c2), max(c1,c2))
        if key not in dsu_store:
            dsu_store[key] = list(range(n))
        return dsu_store[key]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = idx
        for c in range(k):
            # Check adjacent
            ok = True
            for u in adj_sorted[v]:
                if colors[u] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check bicolored cycle
            unions_done = []
            ok2 = True
            for u in adj_sorted[v]:
                if colors[u] >= 0 and colors[u] != c:
                    d = colors[u]
                    dsu = get_dsu(c, d)
                    record = dsu_union(dsu, v, u)
                    if record is None:
                        # Already connected -> cycle
                        # Rollback any unions done so far in this step
                        for rec in reversed(unions_done):
                            k1, k2 = rec[0]
                            dsu_rollback(dsu_store[(min(k1,k2),max(k1,k2))], rec[1])
                        ok2 = False
                        break
                    unions_done.append(((c, d), record))
            
            if not ok2:
                continue
            
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
            
            # Rollback unions
            for rec in reversed(unions_done):
                k1, k2 = rec[0]
                dsu_rollback(dsu_store[(min(k1,k2),max(k1,k2))], rec[1])
        
        return False
    
    if backtrack(0):
        return colors
    return None
