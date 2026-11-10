import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def seed_graph():
        adj = [0]*n
        for k in range(5):
            for a in range(4):
                for b in range(a+1,4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
        return adj
    
    def seed_graph_random():
        adj = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.25:
                    adj[u] |= 1<<v; adj[v] |= 1<<u
        return adj
    
    def edge_count(adj):
        return sum(popcount(a) for a in adj)//2
    
    def build_bad(adj):
        bad = [0]*n
        for u in range(n):
            for w in range(u+1, n):
                if popcount(adj[u] & adj[w]) >= 2:
                    bad[u] |= 1<<w
                    bad[w] |= 1<<u
        return bad
    
    def verify(adj):
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True
    
    def can_add(adj, bad, u, v):
        if (adj[u]>>v)&1: return True
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        nv = adj[v] & ~(1<<u) & ~(1<<v)
        if nv & bad[u]:
            return False
        nu = adj[u] & ~(1<<u) & ~(1<<v)
        if nu & bad[v]:
            return False
        return True
    
    def update_bad_after_add(adj, bad, u, v):
        nv = adj[v] & ~(1<<u) & ~(1<<v)
        nu = adj[u] & ~(1<<u) & ~(1<<v)
        for w in range(n):
            if w == u or w == v: continue
            if (nv >> w) & 1:
                if popcount(adj[u] & adj[w]) >= 2:
                    bad[u] |= 1<<w
                    bad[w] |= 1<<u
            if (nu >> w) & 1:
                if popcount(adj[v] & adj[w]) >= 2:
                    bad[v] |= 1<<w
                    bad[w] |= 1<<v
    
    def update_bad_after_remove(adj, bad, u, v):
        nv = adj[v] & ~(1<<u) & ~(1<<v)
        nu = adj[u] & ~(1<<u) & ~(1<<v)
        for w in range(n):
            if w == u or w == v: continue
            if (nv >> w) & 1:
                if popcount(adj[u] & adj[w]) < 2:
                    bad[u] &= ~(1<<w)
                    bad[w] &= ~(1<<u)
            if (nu >> w) & 1:
                if popcount(adj[v] & adj[w]) < 2:
                    bad[v] &= ~(1<<w)
                    bad[w] &= ~(1<<v)
    
    def do_add(adj, bad, u, v):
        adj[u] |= 1<<v; adj[v] |= 1<<u
        update_bad_after_add(adj, bad, u, v)
    
    def do_remove(adj, bad, u, v):
        adj[u] &= ~(1<<v); adj[v] &= ~(1<<u)
        update_bad_after_remove(adj, bad, u, v)
    
    def local_improve(adj, bad):
        changed = True
        while changed:
            changed = False
            for (u,v) in all_edges:
                if not ((adj[u]>>v)&1) and can_add(adj, bad, u, v):
                    do_add(adj, bad, u, v)
                    changed = True
        return adj, bad
    
    def compute_edge_blocking_scores(adj, bad):
        scores = {}
        for (u,v) in all_edges:
            if (adj[u]>>v)&1:
                continue
            if can_add(adj, bad, u, v):
                continue
            nv = adj[v] & ~(1<<u) & ~(1<<v)
            nu = adj[u] & ~(1<<u) & ~(1<<v)
            score = 0
            if nv & bad[u]:
                score += popcount(nv & bad[u])
            if nu & bad[v]:
                score += popcount(nu & bad[v])
            if popcount(adj[u] & adj[v]) >= 3:
                score += popcount(adj[u] & adj[v])
            if score > 0:
                for w in range(n):
                    if w == u or w == v: continue
                    if ((nv >> w) & 1) and ((bad[u] >> w) & 1):
                        for e in all_edges:
                            if (e[0] == u and e[1] == w) or (e[0] == w and e[1] == u):
                                if (adj[e[0]]>>e[1])&1:
                                    scores[e] = scores.get(e, 0) + 1
                    if ((nu >> w) & 1) and ((bad[v] >> w) & 1):
                        for e in all_edges:
                            if (e[0] == v and e[1] == w) or (e[0] == w and e[1] == v):
                                if (adj[e[0]]>>e[1])&1:
                                    scores[e] = scores.get(e, 0) + 1
        return scores
    
    best = None
    best_count = -1
    rng = random.Random(12345)
    
    for attempt in range(5000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.25:
            adj = seed_graph()
        elif mode < 0.5:
            adj = seed_graph_random()
        else:
            adj = [0]*n
        bad = build_bad(adj)
        order = all_edges[:]
        rng.shuffle(order)
        for (u,v) in order:
            if not ((adj[u]>>v)&1) and can_add(adj, bad, u, v):
                do_add(adj, bad, u, v)
        adj, bad = local_improve(adj, bad)
        cnt = edge_count(adj)
        if not verify(adj):
            continue
        if cnt > best_count:
            best_count = cnt
            best = (adj[:], bad[:])
        if best is not None and attempt % 15 == 0:
            adj2 = best[0][:]
            bad2 = best[1][:]
            edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
            if edges_now:
                blocking = compute_edge_blocking_scores(adj2, bad2)
                if blocking:
                    sorted_edges = sorted(blocking.keys(), key=lambda e: blocking[e], reverse=True)
                    top_k = sorted_edges[:max(1, len(sorted_edges)//3)]
                    u,v = rng.choice(top_k)
                else:
                    u,v = rng.choice(edges_now)
                do_remove(adj2, bad2, u, v)
                adj2, bad2 = local_improve(adj2, bad2)
                cnt2 = edge_count(adj2)
                if verify(adj2) and cnt2 > best_count:
                    best_count = cnt2
                    best = (adj2[:], bad2[:])
    
    if best is None:
        adj = seed_graph()
    else:
        adj = best[0]
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u]>>v)&1:
                res.append([u,v])
    return res
