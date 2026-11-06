import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def has_c4_add(u, v, adj):
        # Check if adding (u, v) creates a C4
        # u, v not adjacent
        nu = adj[u]
        nv = adj[v]
        if len(nu) < len(nv):
            for w in nu:
                # w is neighbor of u
                # Check if w has neighbor in nv
                # If w has neighbor x in nv, and x != w, then C4
                # Note: if w in nv, then u-w-v is triangle.
                # We need x in nv such that w ~ x and x != w.
                for x in adj[w]:
                    if x in nv and x != w:
                        return True
        else:
            for x in nv:
                for w in adj[x]:
                    if w in nu and w != x:
                        return True
        return False

    def has_c4_exist(u, v, adj):
        # Check if (u, v) is in a C4 in current graph
        # u, v adjacent
        nu = adj[u]
        nv = adj[v]
        for w in nu:
            if w == v: continue
            for x in nv:
                if x == u: continue
                if w == x: continue
                if x in adj[w]:
                    return True
        return False

    def try_solve(n, min_deg_target, time_limit):
        adj = [set() for _ in range(n)]
        
        # Initialize
        p = 0.15
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        for step in range(20000):
            if time.time() > time_limit:
                return None
            
            # Check validity
            min_deg = min(len(adj[i]) for i in range(n))
            
            if min_deg >= min_deg_target:
                valid = True
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            if has_c4_exist(i, j, adj):
                                valid = False
                                break
                    if not valid: break
                
                if valid:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Improve
            low_deg = [i for i in range(n) if len(adj[i]) < min_deg_target]
            
            if low_deg:
                u = random.choice(low_deg)
                cands = [v for v in range(n) if v not in adj[u] and v != u]
                random.shuffle(cands)
                added = False
                for v in cands:
                    if not has_c4_add(u, v, adj):
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
                        break
                if not added:
                    # Remove random edge from u
                    if adj[u]:
                        v = random.choice(list(adj[u]))
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                # All degrees ok, but C4 exists
                c4_found = False
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            if has_c4_exist(i, j, adj):
                                adj[i].remove(j)
                                adj[j].remove(i)
                                c4_found = True
                                break
                    if c4_found: break
                if not c4_found:
                    # Should not happen, but remove random edge
                    if adj[0]:
                        j = random.choice(list(adj[0]))
                        adj[0].remove(j)
                        adj[j].remove(0)
        
        return None

    # Try n=46
    res = try_solve(46, 46-39, deadline)
    if res:
        return res
        
    # Try n=45
    res = try_solve(45, 45-39, deadline)
    if res:
        return res
        
    # Try n=44
    res = try_solve(44, 44-39, deadline)
    if res:
        return res
        
    return {'n': 39, 'edges': []}
