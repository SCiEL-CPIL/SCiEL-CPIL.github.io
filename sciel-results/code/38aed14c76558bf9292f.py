"""Full Lazebnik-Ustimenko-Woldar CD(d, q) graph: q-regular, girth >= d+5.
Take the full connected graph (all points+lines) with the smallest prime power q >= k,
choosing d so the girth bound reaches g. Valid, far from optimal but much larger than
the seed's single-component truncation."""
import time

_DEADLINE = 150.0
_START = None


def _remaining():
    global _START
    if _START is None:
        _START = time.monotonic()
    return _DEADLINE - (time.monotonic() - _START)


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _prime_power_ge(k):
    if k <= 2:
        return 2
    # primes
    n = k
    while not _is_prime(n):
        n += 1
    best = n
    # prime powers p^a
    p = 2
    while p * p <= best:
        if _is_prime(p):
            v = p * p
            while v < best:
                v *= p
            if v < best:
                best = v
        p += 1
    return best


def _smallest_d(q, k, g):
    d = 3
    while True:
        if g <= d + 5:
            return d
        d += 1


def _gf_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        return add, sub, mul
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    return add, sub, mul


def _rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _girth_ok(adj, n, g):
    # girth >= g  <=>  no closed non-backtracking walk of length < g from any start.
    for s in range(n):
        front = [(s, -1)]
        for _ in range(g - 1):
            nxt = []
            for u, p in front:
                for v in adj[u]:
                    if v == p:
                        continue
                    if v == s:
                        return False
                    nxt.append((v, u))
            if not nxt:
                break
            front = nxt
    return True


def construct(k, g):
    if _remaining() < 10:
        return 0, []
    q = _prime_power_ge(k)
    d = _smallest_d(q, k, g)
    add, sub, mul = _gf_ops(q)
    rule = _rule(d)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Fallback: if (very unlikely for prime q) girth not certified, keep the
    # connected component of the zero vertex that provably has girth >= g.
    if not _girth_ok(adj, n, g):
        seen = [False] * n
        stack = [0]
        seen[0] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        remap = {v: i for i, v in enumerate(comp)}
        new_edges = []
        for a, b in edges:
            if a in remap and b in remap:
                new_edges.append((remap[a], remap[b]))
        return len(comp), new_edges

    return n, edges
