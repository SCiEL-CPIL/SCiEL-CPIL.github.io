import random
import time

def construct():
    n = 46
    min_deg_req = n - 39  # 7
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    def random_graph(n, p):
        e = []
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    e.append((i,j))
                    adj[i].add(j)
                    adj[j].add(i)
        return e, adj

    def find_c4(adj, n):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u].intersection(adj[v])
                    if len(common) >= 2:
                        clist = list(common)
                        w = clist[0]
                        x = clist[1]
                        return (u, v, w, x)
        return None

    def remove_edge(e, adj, u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        # Remove from list
        if (u,v) in e:
            e.remove((u,v))
        else:
            e.remove((v,u))

    def has_c4_with(u, v, adj):
        nu = adj[u]
        nv = adj[v]
        # Check if there is w in nv, x in nu such that (w,x) in E and w!=x
        for w in nv:
            if w == u: continue
            for x in adj[w]:
                if x in nu and x != w and x != v and x != u:
                    return True
        return False

    best_n = 39
    best_edges = []
    
    for _ in range(50): # Restart
        if time.time() - start_time > time_limit:
            break
            
        e, adj = random_graph(n, 0.4)
        
        # Phase 1: Remove C4s
        c4 = find_c4(adj, n)
        while c4:
            u, v, w, x = c4
            # Remove (u,v)
            remove_edge(e, adj, u, v)
            c4 = find_c4(adj, n)
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break
            
        # Phase 2: Boost degree
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            degs = [len(adj[i]) for i in range(n)]
            min_deg = min(degs)
            if min_deg >= min_deg_req:
                break
            
            u = degs.index(min_deg)
            
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if has_c4_with(u, v, adj):
                    continue
                e.append((u,v))
                adj[u].add(v)
                adj[v].add(u)
                improved = True
                added = True
                break
            
            if not added:
                # Try to remove a random edge to unblock?
                # Or just break.
                break
        
        if min_deg >= min_deg_req:
            return {'n': n, 'edges': [[u,v] for u,v in e]}
            
    return {'n': 39, 'edges': []}
