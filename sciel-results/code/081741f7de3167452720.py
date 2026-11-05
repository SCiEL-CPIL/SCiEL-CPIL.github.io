import random
import time

def construct():
    start = time.time()
    deadline = 145.0
    n = 31
    target_edges = 91

    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            su = adj_sets[u]
            for v in range(u + 1, n):
                if v not in su:
                    c = len(su & adj_sets[v])
                    if c >= 2:
                        total += c * (c - 1) // 2
        return total // 2

    def adj_to_edges(adj_sets):
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # ---- Build the polarity graph of the projective plane of order 5 ----
    q = 5
    pts = []
    seen = set()
    for x in range(q):
        for y in range(q):
            p = (x, y)
            if p not in seen:
                seen.add(p)
                pts.append(p)
    for x in range(q):
        pts.append((x, None))
    pts.append((None, None))
    assert len(pts) == q * q + q + 1 == 31

    def norm(p):
        if p[1] is None:
            return p[0]
        if p[0] is None:
            return 1
        return p[0] * p[0] + p[1] * p[1]

    adj = [set() for _ in range(n)]
    for i in range(n):
        ni = norm(pts[i])
        for j in range(i + 1, n):
            if norm(pts[j]) == ni:
                adj[i].add(j)
                adj[j].add(i)

    # ---- Add the single edge that creates the fewest 4-cycles ----
    best = None
    for i in range(n):
        si = adj[i]
        for j in range(i + 1, n):
            if j not in si:
                c = len(si & adj[j])
                if best is None or c < best[0]:
                    best = (c, i, j)
    adj[best[1]].add(best[2])
    adj[best[2]].add(best[1])

    edges = adj_to_edges(adj)
    assert len(edges) == target_edges

    # ---- Local search: try to reduce the 4-cycle count via swaps ----
    def evaluate_swap(edges, adj_sets, cur_c4, ru, rv, u, v):
        # remove (ru,rv), add (u,v); return new c4 if it is an improvement
        ruN, rvN = adj_sets[ru], adj_sets[rv]
        uN, vN = adj_sets[u], adj_sets[v]
        ruN.discard(rv); rvN.discard(ru)
        uN.add(v); vN.add(u)
        new_c4 = count_c4(adj_sets)
        if new_c4 < cur_c4:
            return new_c4
        uN.discard(v); vN.discard(u)
        ruN.add(rv); rvN.add(ru)
        return None

    cur_c4 = count_c4(adj)
    edge_set = set((i, j) for i in range(n) for j in adj[i] if i < j)

    while time.time() < deadline - 3:
        improved = False
        for _ in range(200):
            if time.time() >= deadline - 3:
                break
            ru, rv = random.choice(sorted(edge_set))
            while True:
                u = random.randrange(n)
                v = random.randrange(n)
                if u < v and (u, v) not in edge_set:
                    break
            new_c4 = evaluate_swap(edges, adj, cur_c4, ru, rv, u, v)
            if new_c4 is not None:
                edge_set.discard((ru, rv)); edge_set.add((u, v))
                edges = adj_to_edges(adj)
                cur_c4 = new_c4
                improved = True
        if not improved:
            break

    return edges
