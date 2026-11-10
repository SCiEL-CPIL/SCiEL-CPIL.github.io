import time
import random

def construct():
    start = time.time()
    deadline = start + 140.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    mask_all = (1 << n) - 1

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

    def build_cn(adj):
        cn = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    cn[u][v] = adj[u] & adj[v]
        return cn

    def edge_count(adj):
        return sum(popcount(a) for a in adj) // 2

    def can_add(adj, cn, u, v):
        if (adj[u] >> v) & 1:
            return True
        return popcount(cn[u][v]) <= 2

    def do_add(adj, cn, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> u) & 1:
                cn[w][v] |= 1 << u
                cn[v][w] |= 1 << u
            if (adj[w] >> v) & 1:
                cn[w][u] |= 1 << v
                cn[u][w] |= 1 << v

    def do_remove(adj, cn, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> u) & 1:
                cn[w][v] &= ~(1 << u)
                cn[v][w] &= ~(1 << u)
            if (adj[w] >> v) & 1:
                cn[w][u] &= ~(1 << v)
                cn[u][w] &= ~(1 << v)

    def verify(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def greedy_fill(adj, cn, order):
        for (u, v) in order:
            if not ((adj[u] >> v) & 1) and can_add(adj, cn, u, v):
                do_add(adj, cn, u, v)

    def local_improve(adj, cn):
        changed = True
        while changed:
            changed = False
            for (u, v) in all_edges:
                if not ((adj[u] >> v) & 1) and can_add(adj, cn, u, v):
                    do_add(adj, cn, u, v)
                    changed = True
        return adj, cn

    best = [None, None]
    best_count = [-1]
    rng = random.Random(12345)

    def record(adj, cn):
        cnt = edge_count(adj)
        if cnt > best_count[0]:
            best_count[0] = cnt
            best[0] = adj[:]
            best[1] = [row[:] for row in cn]
        return cnt

    def two_opt(adj, cn):
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            edges_now = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
            for (u, v) in edges_now:
                if time.time() > deadline - 10.0:
                    break
                do_remove(adj, cn, u, v)
                greedy_fill(adj, cn, all_edges)
                cnt = edge_count(adj)
                if cnt > best_count[0]:
                    best_count[0] = cnt
                    best[0] = adj[:]
                    best[1] = [row[:] for row in cn]
                    improved = True
                else:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (adj[w] >> u) & 1:
                            cn[w][v] |= 1 << u
                            cn[v][w] |= 1 << u
                        if (adj[w] >> v) & 1:
                            cn[w][u] |= 1 << v
                            cn[u][w] |= 1 << v
                    break
        return adj, cn

    def swap_move(adj, cn):
        # remove one edge, add another, re-greedy; keep if it grows.
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            edges_now = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
            for (u, v) in edges_now:
                if time.time() > deadline - 10.0:
                    break
                do_remove(adj, cn, u, v)
                for (a, b) in all_edges:
                    if not ((adj[a] >> b) & 1) and can_add(adj, cn, a, b):
                        do_add(adj, cn, a, b)
                        greedy_fill(adj, cn, all_edges)
                        cnt = edge_count(adj)
                        if cnt > best_count[0]:
                            best_count[0] = cnt
                            best[0] = adj[:]
                            best[1] = [row[:] for row in cn]
                            improved = True
                        break
                if not improved:
                    # restore removed edge
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (adj[w] >> u) & 1:
                            cn[w][v] |= 1 << u
                            cn[v][w] |= 1 << u
                        if (adj[w] >> v) & 1:
                            cn[w][u] |= 1 << v
                            cn[u][w] |= 1 << v
                    break
        return adj, cn

    for attempt in range(200000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.25:
            adj = seed_graph()
        else:
            adj = [0] * n
        cn = build_cn(adj)
        order = all_edges[:]
        rng.shuffle(order)
        greedy_fill(adj, cn, order)
        adj, cn = local_improve(adj, cn)
        record(adj, cn)

        # periodic perturbation from the best
        if best[0] is not None and attempt % 20 == 0 and time.time() < deadline - 8.0:
            adj2 = best[0][:]
            cn2 = [row[:] for row in best[1]]
            edges_now = [(u, v) for (u, v) in all_edges if (adj2[u] >> v) & 1]
            if edges_now:
                u, v = rng.choice(edges_now)
                do_remove(adj2, cn2, u, v)
                order2 = all_edges[:]
                rng.shuffle(order2)
                greedy_fill(adj2, cn2, order2)
                adj2, cn2 = local_improve(adj2, cn2)
                record(adj2, cn2)

    # final polish on the best graph
    if best[0] is not None:
        adj = best[0][:]
        cn = [row[:] for row in best[1]]
        adj, cn = local_improve(adj, cn)
        record(adj, cn)
        if time.time() < deadline - 8.0:
            adj, cn = two_opt(adj, cn)
            record(adj, cn)
        if time.time() < deadline - 5.0:
            adj, cn = swap_move(adj, cn)
            record(adj, cn)

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
