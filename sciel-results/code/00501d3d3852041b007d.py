import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = float('inf')
    best_edges = None
    
    def verify_graph(n, edges, k, g):
        if n == 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        max_depth = (g - 1) // 2
        for u in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[u] = 0
            q = deque([(u, -1)])
            while q:
                c, p = q.popleft()
                if dist[c] >= max_depth:
                    continue
                for v in adj[c]:
                    if v == p:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[c] + 1
                        par[v] = c
                        q.append((v, c))
                    else:
                        if v == par[c]:
                            continue
                        if dist[c] + dist[v] + 1 < g:
                            return False
        return True

    def build_luw_graph(q_val, d_val):
        if q_val == 4:
            tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: tbl[x][y]
        else:
            add = lambda x,y: (x+y)%q_val
            sub = lambda x,y: (x-y)%q_val
            mul = lambda x,y: x*y%q_val
        rules = {2:(1,1),3:(2,1),4:(1,2)}
        for c in range(5, d_val+1):
            rc = (c-5)%4
            c0 = c-(c-5)%4
            if rc==0: rules[c]=(1,c0-2)
            elif rc==1: rules[c]=(c0-1,1)
            elif rc==2: rules[c]=(c0,1)
            else: rules[c]=(1,c0+1)
        def nbrs(vtx):
            s, x = vtx
            for f in range(q_val):
                y = [None,f]+[0]*(d_val-1)
                for c in range(2,d_val+1):
                    a,b = rules[c]
                    if s==0: y[c]=add(x[c],mul(y[a],x[b]))
                    else: y[c]=sub(x[c],mul(x[a],y[b]))
                yield 1-s,tuple(y)
        st=(0,(None,)+(0,)*d_val)
        idx={st:0}; ord_list=[st]; edges=[]
        for i,v in enumerate(ord_list):
            for o in nbrs(v):
                if o not in idx:
                    idx[o]=len(ord_list)
                    ord_list.append(o)
                if v[0]==0:
                    edges.append((idx[v],idx[o]))
        return len(ord_list),edges

    # Try to find smallest valid graph
    min_d = max(3, (g-2)//2)
    for d_val in range(min_d, min_d+25):
        if time.time() > deadline - 15:
            break
        try:
            n, edges = build_luw_graph(k, d_val)
            if n >= best_n:
                continue
            if verify_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
                break
        except Exception:
            pass

    if best_edges is None:
        return 0, []
    return best_n, best_edges
