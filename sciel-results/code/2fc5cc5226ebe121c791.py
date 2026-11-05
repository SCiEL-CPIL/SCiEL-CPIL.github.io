"""K2,3-free graph on 20 vertices: maximize edges via randomized greedy + polish."""
import random
import time

N = 20


def _common_count(adj, u, v):
    # number of common neighbours of u and v (adjacency sets)
    if u in adj[v]:
        return len(adj[u] & adj[v])
    return len(adj[u] & adj[v])


def _valid_add(adj, u, v):
    # can we add edge (u,v) without creating a K2,3?
    # after adding, u and v become common neighbours of each other.
    # For any w != u,v: common(u,w) and common(v,w) may increase by 1
    #   if the other is adjacent to w.
    su = adj[u]
    sv = adj[v]
    for w in su:
        if w == v:
            continue
        # common(u,w) currently includes v? no (u-v not edge yet)
        # after add, v becomes a neighbour of u, so common(u,w) += 1 iff w in sv
        if w in sv:
            c = len(su & adj[w])  # common(u,w) before; v not in su so not counted
            if c + 1 >= 3:
                return False
    for w in sv:
        if w == u:
            continue
        if w in su:
            continue  # handled above
        # common(v,w) increases by 1 iff w in su
        if w in su:
            c = len(sv & adj[w])
            if c + 1 >= 3:
                return False
    # also check common(u,v) after add
    if len(su & sv) >= 3:
        return False
    return True


def _try_add(adj, u, v):
    if u == v or v in adj[u]:
        return False
    if not _valid_add(adj, u, v):
        return False
    adj[u].add(v)
    adj[v].add(u)
    return True


def _greedy(adj, order, rng):
    # add edges in given order if valid
    cnt = 0
    for (u, v) in order:
        if v not in adj[u] and _valid_add(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            cnt += 1
    return cnt


def _random_order(rng):
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(pairs)
    return pairs


def _polish(adj, rng, max_iter=4000):
    # try to add any missing edge; if none, try removals to enable additions
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # addition pass
        for u in range(N):
            for v in range(u + 1, N):
                if v not in adj[u]:
                    if _valid_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
        # removal pass: remove a random edge, see if we can gain net
        if not improved:
            edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            if not edges:
                break
            u, v = rng.choice(edges)
            adj[u].discard(v)
            adj[v].discard(u)
            before = 1
            # try to add back anything
            for a in range(N):
                for b in range(a + 1, N):
                    if b not in adj[a] and _valid_add(adj, a, b):
                        adj[a].add(b)
                        adj[b].add(a)
                        before += 1
            if before <= 0:
                # net loss, revert
                adj[u].add(v)
                adj[v].add(u)
                improved = True  # force another attempt
    return adj


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _to_edges(adj):
    return [[u, v] for u in range(N) for v in range(u + 1, N) if v in adj[u]]


def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    rng = random.Random(12345)
    best_adj = None
    best_count = -1

    while time.time() < deadline:
        adj = [set() for _ in range(N)]
        order = _random_order(rng)
        _greedy(adj, order, rng)
        _polish(adj, rng, max_iter=1500)
        c = _edge_count(adj)
        if c > best_count:
            best_count = c
            best_adj = [set(s) for s in adj]
            if best_count >= 170:
                break

    if best_adj is None:
        # fallback: K4 union (always valid)
        best_adj = [set() for _ in range(N)]
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    best_adj[u].add(v)
                    best_adj[v].add(u)
    return _to_edges(best_adj)
