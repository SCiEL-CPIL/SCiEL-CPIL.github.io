import time
import random

# EVOLVE-BLOCK-START

def _is_3_colorable(adj):
    """Exact 3-colorability via backtracking, symmetry breaking, MRV ordering."""
    n = len(adj)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    adj_pos = [sorted(adj[v], key=lambda u: pos[u]) for v in range(n)]

    def dfs(i, color):
        if i == n:
            return True
        v = order[i]
        forbidden = 0
        for u in adj_pos[v]:
            c = color[u]
            if c >= 0:
                forbidden |= (1 << c)
        avail = (~forbidden) & 0b111
        if pos[v] == 0:
            avail &= 0b001
        while avail:
            lsb = avail & (-avail)
            avail -= lsb
            c = lsb.bit_length() - 1
            color[v] = c
            if dfs(i + 1, color):
                return True
            color[v] = -1
        return False

    color = [-1] * n
    return dfs(0, color)


def _is_triangle_free(adj):
    for u in range(len(adj)):
        au = adj[u]
        lst = list(au)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[j] in lst[i]:
                    return False
    return True


def _edge_count(adj):
    return sum(len(a) for a in adj) // 2


def _valid(adj):
    return _is_triangle_free(adj) and not _is_3_colorable(adj)


def construct():
    start = time.time()
    deadline = start + 135.0
    n = 16
    rng = random.Random(20240501)

    def remaining():
        return deadline - time.time()

    def greedy_build(seed):
        r = random.Random(seed)
        adj = [set() for _ in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        r.shuffle(pairs)
        for u, v in pairs:
            if v in adj[u]:
                continue
            if adj[u] & adj[v]:
                continue
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def polish(adj, time_limit):
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            for u, v in pairs:
                if v in adj[u]:
                    continue
                if adj[u] & adj[v]:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                if not _is_3_colorable(adj):
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        return adj

    best_adj = None
    best_count = -1

    for seed in range(60):
        if remaining() < 25.0:
            break
        adj = greedy_build(seed)
        if not _is_triangle_free(adj):
            continue
        # polish to add more edges while staying 4-chromatic
        adj = polish(adj, max(0.0, remaining() - 25.0) / 3.0)
        if _is_triangle_free(adj) and not _is_3_colorable(adj):
            cnt = _edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = [set(a) for a in adj]

    if best_adj is None:
        # Fallback: Götzsch graph (known valid, 10 edges)
        adj = [set() for _ in range(n)]
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, w in cycle:
            adj[u].add(w); adj[w].add(u)
        for u, w in cycle:
            adj[u].add(5 + w); adj[5 + w].add(u)
            adj[w].add(5 + u); adj[5 + u].add(w)
        for u in range(5):
            adj[5 + u].add(10); adj[10].add(5 + u)
        best_adj = adj

    # Final safety: if invalid, use Götzsch
    if not _valid(best_adj):
        adj = [set() for _ in range(n)]
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, w in cycle:
            adj[u].add(w); adj[w].add(u)
        for u, w in cycle:
            adj[u].add(5 + w); adj[5 + w].add(u)
            adj[w].add(5 + u); adj[5 + u].add(w)
        for u in range(5):
            adj[5 + u].add(10); adj[10].add(5 + u)
        best_adj = adj

    edges = [[u, v] for u in range(n) for v in best_adj[u] if u < v]
    return edges

# EVOLVE-BLOCK-END
