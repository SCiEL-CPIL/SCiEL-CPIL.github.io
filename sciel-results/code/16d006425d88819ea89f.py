import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Bitmask-based feasibility tracking:
    # For each vertex u, bad[u] is a bitmask of vertices v such that adding edge (u,v) 
    # would create a K2,3. Specifically, bad[u] contains v if:
    #   - common[u][v] >= 3, OR
    #   - there exists w with adj[v] has w and common[u][w] >= 2, OR
    #   - there exists w with adj[u] has w and common[v][w] >= 2
    # We maintain bad[] incrementally.
    
    def seed_graph():
        adj = [0]*n
        common = [[0]*n for _ in range(n)]
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
                    for w in range(n):
                        if w == u or w == v: continue
                        if (adj[v]>>w)&1:
                            common[u][w]+=1; common[w][u]+=1
                        if (adj[u]>>w)&1:
                            common[v][w]+=1; common[w][v]+=1
        # Build bad from scratch
        bad = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u]>>v)&1: continue
                feasible = True
                if common[u][v] >= 3:
                    feasible = False
                else:
                    # Check all w
                    for w in range(n):
                        if w == u or w == v: continue
                        if (adj[v]>>w)&1 and common[u][w] >= 2:
                            feasible = False; break
                        if (adj[u]>>w)&1 and common[v][w] >= 2:
                            feasible = False; break
                if not feasible:
                    bad[u] |= 1<<v
                    bad[v] |= 1<<u
        return adj, common, bad
    
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj)//2
    
    def can_add(adj, bad, u, v):
        if (adj[u]>>v)&1: return True
        return not ((bad[u]>>v)&1)
    
    def do_add(adj, common, bad, u, v):
        """Add edge (u,v) and update bad for all affected pairs."""
        adj[u] |= 1<<v; adj[v] |= 1<<u
        
        # After adding (u,v), the common neighbor counts change:
        # For each w in adj[u] (excluding v): common[v][w] += 1, common[w][v] += 1
        # For each w in adj[v] (excluding u): common[u][w] += 1, common[w][u] += 1
        # This can make some previously-feasible edges infeasible.
        
        # The pairs that could become infeasible are:
        # 1. (u, x) for x not in adj[u]: common[u][x] may have increased (if x was in adj[v])
        # 2. (v, x) for x not in adj[v]: common[v][x] may have increased (if x was in adj[u])
        # 3. (x, y) where x in adj[u], y in adj[v]: common[x][y] may have increased
        # 4. (u, v) itself is now an edge, so no longer relevant for bad
        
        # Let's recompute bad entries that could be affected.
        # Affected vertices: u, v, and all neighbors of u and v.
        
        # First, update common counts
        au = adj[u] & ~(1<<v)
        av = adj[v] & ~(1<<u)
        for w in range(n):
            if (au>>w)&1:
                common[v][w]+=1; common[w][v]+=1
            if (av>>w)&1:
                common[u][w]+=1; common[w][u]+=1
        
        # Now update bad. The pairs whose feasibility might change:
        # For any pair (a,b) not already an edge, if a or b is u or v, or if 
        # common[a][b] changed (i.e., a in N(u) and b in N(v), etc.)
        
        # Simplest correct approach: recompute bad for all pairs involving u or v,
        # and for pairs (x,y) where x in N(u) and y in N(v).
        
        # Let's just recompute bad for all pairs that could be affected.
        # A pair (a,b) is affected if:
        # - a==u or b==u or a==v or b==v, OR
        # - a in N(u) and b in N(v), OR
        # - a in N(v) and b in N(u)
        
        # Collect affected pairs
        affected = set()
        for x in range(n):
            if x == u or x == v: continue
            if (adj[u]>>x)&1:
                # (u,x) is now an edge, but (x,y) for y in N(v) might be affected
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (adj[v]>>y)&1:
                        affected.add((min(x,y), max(x,y)))
            if (adj[v]>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (adj[u]>>y)&1:
                        affected.add((min(x,y), max(x,y)))
        
        # Also pairs involving u or v with non-neighbors
        for x in range(n):
            if x == u or x == v: continue
            if not ((adj[u]>>x)&1):
                affected.add((min(u,x), max(u,x)))
            if not ((adj[v]>>x)&1):
                affected.add((min(v,x), max(v,x)))
        
        # For each affected pair, recompute feasibility
        for (a,b) in affected:
            if (adj[a]>>b)&1:
                # Already an edge, clear bad bits
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
                continue
            # Check if adding (a,b) would be feasible
            feasible = True
            if common[a][b] >= 3:
                feasible = False
            else:
                for w in range(n):
                    if w == a or w == b: continue
                    if (adj[b]>>w)&1 and common[a][w] >= 2:
                        feasible = False; break
                    if (adj[a]>>w)&1 and common[b][w] >= 2:
                        feasible = False; break
            if not feasible:
                bad[a] |= 1<<b
                bad[b] |= 1<<a
            else:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
    
    def do_remove(adj, common, bad, u, v):
        """Remove edge (u,v) and update bad for all affected pairs."""
        au = adj[u] & ~(1<<v)
        av = adj[v] & ~(1<<u)
        
        # Update common counts (decrement)
        for w in range(n):
            if (au>>w)&1:
                common[v][w]-=1; common[w][v]-=1
            if (av>>w)&1:
                common[u][w]-=1; common[w][u]-=1
        
        adj[u] &= ~(1<<v); adj[v] &= ~(1<<u)
        
        # Affected pairs: same logic as do_add
        affected = set()
        for x in range(n):
            if x == u or x == v: continue
            if (au>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (av>>y)&1:
                        affected.add((min(x,y), max(x,y)))
            if (av>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (au>>y)&1:
                        affected.add((min(x,y), max(x,y)))
        
        for x in range(n):
            if x == u or x == v: continue
            if not ((au>>x)&1):
                affected.add((min(u,x), max(u,x)))
            if not ((av>>x)&1):
                affected.add((min(v,x), max(v,x)))
        
        # The pair (u,v) itself is now a non-edge
        affected.add((min(u,v), max(u,v)))
        
        for (a,b) in affected:
            if (adj[a]>>b)&1:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
                continue
            feasible = True
            if common[a][b] >= 3:
                feasible = False
            else:
                for w in range(n):
                    if w == a or w == b: continue
                    if (adj[b]>>w)&1 and common[a][w] >= 2:
                        feasible = False; break
                    if (adj[a]>>w)&1 and common[b][w] >= 2:
                        feasible = False; break
            if not feasible:
                bad[a] |= 1<<b
                bad[b] |= 1<<a
            else:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
    
    def local_improve(adj, common, bad):
        changed = True
        while changed:
            changed = False
            for (u,v) in all_edges:
                if not ((adj[u]>>v)&1) and not ((bad[u]>>v)&1):
                    do_add(adj, common, bad, u, v)
                    changed = True
        return adj, common, bad
    
    best = None
    best_count = -1
    rng = random.Random(12345)
    
    for attempt in range(50000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.3:
            adj, common, bad = seed_graph()
        else:
            adj = [0]*n
            common = [[0]*n for _ in range(n)]
            bad = [0]*n
        
        order = all_edges[:]
        rng.shuffle(order)
        for (u,v) in order:
            if not ((adj[u]>>v)&1) and not ((bad[u]>>v)&1):
                do_add(adj, common, bad, u, v)
        
        adj, common, bad = local_improve(adj, common, bad)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best = (adj[:], [row[:] for row in common], bad[:])
        
        if best is not None and attempt % 50 == 0:
            adj2, common2, bad2 = best[0][:], [row[:] for row in best[1]], best[2][:]
            edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
            if edges_now:
                u,v = rng.choice(edges_now)
                do_remove(adj2, common2, bad2, u, v)
                adj2, common2, bad2 = local_improve(adj2, common2, bad2)
                cnt2 = edge_count(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best = (adj2[:], [row[:] for row in common2], bad2[:])
    
    if best is None:
        adj, common, bad = seed_graph()
    else:
        adj = best[0]
    
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u]>>v)&1:
                res.append([u,v])
    return res
