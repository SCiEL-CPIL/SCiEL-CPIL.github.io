import time
import random

def construct():
    start = time.time()
    deadline = start + 145.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]

    def popcount(x):
        return bin(x).count('1')

    def build_bad(adj):
        bad = [0]*n
        for u in range(n):
            for w in range(n):
                if w == u:
                    continue
                if popcount(adj[u] & adj[w]) >= 2:
                    bad[u] |= 1 << w
        return bad

    def seed_graph():
        adj = [0]*n
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
        return adj

    def edge_count(adj):
        return sum(popcount(a) for a in adj)//2

    def can_add(adj, bad, u, v):
        if (adj[u]>>v)&1:
            return True
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
            if w == u or w == v:
                continue
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
            if w == u or w == v:
                continue
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
        update_bad_after_remove(adj, bad, u, v)
        adj[u] &= ~(1<<v); adj[v] &= ~(1<<u)

    def local_improve(adj, bad):
        changed = True
        while changed:
            changed = False
            for (u, v) in all_edges:
                if time.time() > deadline - 2.0:
                    return adj, bad
                if not ((adj[u]>>v)&1) and can_add(adj, bad, u, v):
                    do_add(adj, bad, u, v)
                    changed = True
        return adj, bad

    def hill_climb(adj, bad):
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for (u1, v1) in all_edges:
                if time.time() > deadline - 2.0:
                    return adj, bad
                if ((adj[u1]>>v1)&1):
                    do_remove(adj, bad, u1, v1)
                    added = False
                    for (u2, v2) in all_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if not ((adj[u2]>>v2)&1) and can_add(adj, bad, u2, v2):
                            do_add(adj, bad, u2, v2)
                            added = True
                            changed = True
                            break
                    if not added:
                        do_add(adj, bad, u1, v1)
        return adj, bad

    best = None
    best_count = -1
    rng = random.Random(12345)

    # Phase 1: Random greedy + local improvement (fast thanks to bad bitmasks)
    for attempt in range(8000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.3:
            adj = seed_graph()
        else:
            adj = [0]*n
        bad = build_bad(adj)
        order = all_edges[:]
        rng.shuffle(order)
        for (u, v) in order:
            if time.time() > deadline - 2.0:
                break
            if not ((adj[u]>>v)&1) and can_add(adj, bad, u, v):
                do_add(adj, bad, u, v)
        adj, bad = local_improve(adj, bad)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best = (adj[:], bad[:])
        # light perturbation of the incumbent periodically
        if best is not None and attempt % 15 == 0:
            adj2, bad2 = best[0][:], best[1][:]
            edges_now = [(u, v) for (u, v) in all_edges if (adj2[u]>>v)&1]
            if edges_now:
                u, v = rng.choice(edges_now)
                do_remove(adj2, bad2, u, v)
                adj2, bad2 = local_improve(adj2, bad2)
                cnt2 = edge_count(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best = (adj2[:], bad2[:])

    # Phase 2: Perturbation + local improvement + hill climb
    if best is not None:
        for attempt in range(3000):
            if time.time() > deadline - 3.0:
                break
            adj2, bad2 = best[0][:], best[1][:]
            k = rng.randint(1, 5)
            edges_now = [(u, v) for (u, v) in all_edges if (adj2[u]>>v)&1]
            if not edges_now:
                break
            for _ in range(k):
                if not edges_now:
                    break
                u, v = rng.choice(edges_now)
                do_remove(adj2, bad2, u, v)
                edges_now = [(u, v) for (u, v) in all_edges if (adj2[u]>>v)&1]
            adj2, bad2 = local_improve(adj2, bad2)
            adj2, bad2 = hill_climb(adj2, bad2)
            cnt2 = edge_count(adj2)
            if cnt2 > best_count:
                best_count = cnt2
                best = (adj2[:], bad2[:])

    # Phase 3: Aggressive hill climb on the incumbent
    if best is not None:
        adj, bad = best[0][:], best[1][:]
        for _ in range(10):
            if time.time() > deadline - 1.0:
                break
            adj, bad = hill_climb(adj, bad)
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best = (adj[:], bad[:])

    if best is None:
        adj = seed_graph()
    else:
        adj = best[0]
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u]>>v)&1:
                res.append([u, v])
    return res
