import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def popcount(x):
        return bin(x).count('1')

    def seed_graph():
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def edge_count(adj):
        return sum(popcount(a) for a in adj) // 2

    def build_bad(adj):
        bad = [0] * n
        for u in range(n):
            for w in range(n):
                if w == u:
                    continue
                if popcount(adj[u] & adj[w]) >= 2:
                    bad[u] |= 1 << w
        return bad

    def verify(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def can_add(adj, bad, u, v):
        if (adj[u] >> v) & 1:
            return True
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        nv = adj[v] & ~(1 << u) & ~(1 << v)
        if nv & bad[u]:
            return False
        nu = adj[u] & ~(1 << u) & ~(1 << v)
        if nu & bad[v]:
            return False
        return True

    def update_bad_after_add(adj, bad, u, v):
        nv = adj[v] & ~(1 << u) & ~(1 << v)
        nu = adj[u] & ~(1 << u) & ~(1 << v)
        for w in range(n):
            if w == u or w == v:
                continue
            if (nv >> w) & 1:
                if popcount(adj[u] & adj[w]) >= 2:
                    bad[u] |= 1 << w
                    bad[w] |= 1 << u
            if (nu >> w) & 1:
                if popcount(adj[v] & adj[w]) >= 2:
                    bad[v] |= 1 << w
                    bad[w] |= 1 << v

    def update_bad_after_remove(adj, bad, u, v):
        nv = adj[v] & ~(1 << u) & ~(1 << v)
        nu = adj[u] & ~(1 << u) & ~(1 << v)
        for w in range(n):
            if w == u or w == v:
                continue
            if (nv >> w) & 1:
                if popcount(adj[u] & adj[w]) < 2:
                    bad[u] &= ~(1 << w)
                    bad[w] &= ~(1 << u)
            if (nu >> w) & 1:
                if popcount(adj[v] & adj[w]) < 2:
                    bad[v] &= ~(1 << w)
                    bad[w] &= ~(1 << v)

    def do_add(adj, bad, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        update_bad_after_add(adj, bad, u, v)

    def do_remove(adj, bad, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        update_bad_after_remove(adj, bad, u, v)

    def greedy_fill(adj, bad, order):
        for (u, v) in order:
            if not ((adj[u] >> v) & 1) and can_add(adj, bad, u, v):
                do_add(adj, bad, u, v)

    def local_improve(adj, bad):
        changed = True
        while changed:
            changed = False
            for (u, v) in all_edges:
                if not ((adj[u] >> v) & 1) and can_add(adj, bad, u, v):
                    do_add(adj, bad, u, v)
                    changed = True
        return adj, bad

    def swap_polish(adj, bad):
        # 2-opt style: remove one existing edge, re-greedy, keep if graph grows.
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            edges_now = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
            for (u, v) in edges_now:
                if time.time() > deadline - 10.0:
                    break
                do_remove(adj, bad, u, v)
                greedy_fill(adj, bad, all_edges)
                cnt = edge_count(adj)
                if cnt > best_count[0]:
                    best_count[0] = cnt
                    best[0] = adj[:]
                    best[1] = bad[:]
                    improved = True
                else:
                    do_remove(adj, bad, u, v) if False else None
                    # restore: re-add u,v (it is always feasible to re-add the removed edge
                    # because the graph without it is a subgraph and adding it back is safe
                    # only if it was feasible before; re-greedy already maximised, so just
                    # re-add to restore the previous state for the next trial).
                    # Re-add the removed edge directly (it was present before removal).
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    update_bad_after_add(adj, bad, u, v)
                    # re-greedy to recover any edges that greedy_fill dropped during the trial
                    greedy_fill(adj, bad, all_edges)
                    break
        return adj, bad

    best = [None, None]
    best_count = [-1]
    rng = random.Random(12345)

    for attempt in range(20000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.3:
            adj = seed_graph()
        else:
            adj = [0] * n
        bad = build_bad(adj)
        order = all_edges[:]
        rng.shuffle(order)
        greedy_fill(adj, bad, order)
        adj, bad = local_improve(adj, bad)
        cnt = edge_count(adj)
        if not verify(adj):
            continue
        if cnt > best_count[0]:
            best_count[0] = cnt
            best[0] = adj[:]
            best[1] = bad[:]
        # periodic perturbation: drop a random edge from the best, re-greedy
        if best[0] is not None and attempt % 15 == 0 and time.time() < deadline - 8.0:
            adj2 = best[0][:]
            bad2 = best[1][:]
            edges_now = [(u, v) for (u, v) in all_edges if (adj2[u] >> v) & 1]
            if edges_now:
                u, v = rng.choice(edges_now)
                do_remove(adj2, bad2, u, v)
                order2 = all_edges[:]
                rng.shuffle(order2)
                greedy_fill(adj2, bad2, order2)
                adj2, bad2 = local_improve(adj2, bad2)
                cnt2 = edge_count(adj2)
                if verify(adj2) and cnt2 > best_count[0]:
                    best_count[0] = cnt2
                    best[0] = adj2[:]
                    best[1] = bad2[:]

    # final high-precision polish pass on the best graph
    if best[0] is not None:
        adj = best[0][:]
        bad = best[1][:]
        adj, bad = local_improve(adj, bad)
        cnt = edge_count(adj)
        if verify(adj) and cnt > best_count[0]:
            best_count[0] = cnt
            best[0] = adj[:]
            best[1] = bad[:]
        swap_polish(adj, bad)

    if best[0] is None:
        adj = seed_graph()
    else:
        adj = best[0]

    res = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                res.append([u, v])
    return res
