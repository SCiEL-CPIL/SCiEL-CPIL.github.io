import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _deadline():
        return time.time() > DEADLINE

    def _make_adj(edges):
        adj = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def _is_triangle_free(adj):
        for (u, v) in all_edges:
            if (adj[u] >> v) & 1:
                if (adj[u] & adj[v]) != 0:
                    return False
        return True

    def _is_3_colorable(adj):
        color = [-1] * n
        nb = [[] for _ in range(n)]
        for u in range(n):
            a = adj[u]
            while a:
                lsb = a & (-a)
                v = lsb.bit_length() - 1
                nb[u].append(v)
                a ^= lsb

        def backtrack():
            best_v = -1
            best_avail = 4
            for v in range(n):
                if color[v] != -1:
                    continue
                used = 0
                for u in nb[v]:
                    if color[u] != -1:
                        used |= (1 << color[u])
                avail = 3 - bin(used).count('1')
                if avail == 0:
                    return False
                if avail < best_avail:
                    best_avail = avail
                    best_v = v
                    if avail == 1:
                        break
            if best_v == -1:
                return True
            used = 0
            for u in nb[best_v]:
                if color[u] != -1:
                    used |= (1 << color[u])
            for c in range(3):
                if not (used >> c) & 1:
                    color[best_v] = c
                    if backtrack():
                        return True
                    color[best_v] = -1
            return False

        return backtrack()

    def _greedy_triangle_free(order, start_edges=None):
        adj = [0] * n
        edges = []
        if start_edges:
            for (u, v) in start_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        for (u, v) in order:
            if (adj[u] >> v) & 1:
                continue
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return edges, adj

    def _groetzsch_edges():
        # Mycielski of C5 = Grötzsch graph, 11 vertices, 20 edges, 4-chromatic, triangle-free
        e = []
        # C5
        for i in range(5):
            e.append((i, (i + 1) % 5))
        # Mycielski copies 5..9
        for i in range(5):
            e.append((5 + i, i))
            e.append((5 + i, (i + 1) % 5))
        # Center 10
        for i in range(5):
            e.append((10, 5 + i))
        return e

    best_edges = None
    best_count = -1
    rng = random.Random(12345)

    def _try_candidate(edges):
        nonlocal best_edges, best_count
        if not edges:
            return
        adj = _make_adj(edges)
        if not _is_triangle_free(adj):
            return
        if _is_3_colorable(adj):
            return
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges[:]

    # Phase A: random greedy from empty
    attempts = 0
    while not _deadline() and attempts < 30000:
        attempts += 1
        order = all_edges[:]
        rng.shuffle(order)
        edges, adj = _greedy_triangle_free(order)
        _try_candidate(edges)

    # Phase B: random greedy from Grötzsch core
    groetzsch = _groetzsch_edges()
    while not _deadline() and attempts < 60000:
        attempts += 1
        order = all_edges[:]
        rng.shuffle(order)
        edges, adj = _greedy_triangle_free(order, start_edges=groetzsch)
        _try_candidate(edges)

    # Phase C: local search - edge swaps to grow
    if best_edges is not None:
        for _ in range(80000):
            if _deadline():
                break
            cur = best_edges[:]
            adj = _make_adj(cur)
            cur_set = set(cur)
            if not cur:
                break
            u, v = rng.choice(cur)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            cur.remove((u, v))
            cur_set.discard((u, v))
            improved = False
            for _ in range(30):
                a, b = rng.choice(all_edges)
                if (a, b) in cur_set:
                    continue
                if (adj[a] >> b) & 1:
                    continue
                if (adj[a] & adj[b]) == 0:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    cur.append((a, b))
                    cur_set.add((a, b))
                    improved = True
                    break
            if not improved:
                continue
            if _is_triangle_free(adj) and not _is_3_colorable(adj):
                if len(cur) > best_count:
                    best_count = len(cur)
                    best_edges = cur[:]

    if best_edges is not None:
        adj = _make_adj(best_edges)
        if _is_triangle_free(adj) and not _is_3_colorable(adj):
            return [[u, v] for (u, v) in best_edges]

    # Fallback: Grötzsch graph (isolated vertices 11..15)
    return [[u, v] for (u, v) in groetzsch]
