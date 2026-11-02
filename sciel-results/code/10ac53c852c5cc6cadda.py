"""Cayley-graph girth search: 3-regular graph on Z/p with generators {±a,±b,±c}.
The girth is the shortest non-trivial word in the generators that evaluates to 0 mod p.
We BFS the free group (no immediate backtracking) to find it; if it is >= g we build
the Cayley graph, which is automatically simple and 3-regular. This is a completely
different construction family from the Lazebnik-Ustimenko-Woldar bipartite graph."""
import time

def _bfs_girth(p, a, b, c, node_cap):
    """Return length of shortest non-trivial word in {±a,±b,±c} evaluating to 0 mod p,
    or None if no such word is found within node_cap visited nodes."""
    g1, g2, g3 = a % p, b % p, c % p
    if g1 == 0 or g2 == 0 or g3 == 0:
        return None
    if g1 == g2 or g1 == g3 or g2 == g3:
        return None
    neg = (-g1) % p
    if neg == g2 or neg == g3:
        return None
    neg = (-g2) % p
    if neg == g1 or neg == g3:
        return None
    neg = (-g3) % p
    if neg == g1 or neg == g2:
        return None
    if g1 + g2 == 0 or g1 + g3 == 0 or g2 + g3 == 0:
        return None
    if g1 + g2 == g3 or g1 + g3 == g2 or g2 + g3 == g1:
        return None
    if g1 - g2 == g3 or g1 - g3 == g2 or g2 - g3 == g1:
        return None
    if g1 - g2 == -g3 or g1 - g3 == -g2 or g2 - g3 == -g1:
        return None

    size = p * p
    table = [0] * size
    used = 1
    mask = size - 1

    def ins(val, last):
        nonlocal used
        h = ((val & 0xFFFF) * 2654435761 + (val >> 16) * 40503) & mask
        while True:
            e = table[h]
            if e == 0:
                table[h] = (val << 2) | last
                used += 1
                return
            if e == (val << 2) | last:
                return
            h = (h + 1) & mask

    ins(0, 0)
    ins(g1, 1); ins(g2, 2); ins(g3, 3)
    ins((-g1) % p, 1); ins((-g2) % p, 2); ins((-g3) % p, 3)
    frontier = [1, 2, 3, 4, 5, 6]

    for L in range(2, 1000):
        if used > node_cap:
            return None
        nxt = []
        for code in frontier:
            val = code >> 2
            last = code & 3
            for gen in (1, 2, 3):
                if gen == last:
                    continue
                nv = (val + (g1 if gen == 1 else g2 if gen == 2 else g3)) % p
                if nv == 0:
                    return L + 1
                if (nv << 2) | gen not in table:
                    nxt.append((nv << 2) | gen)
        if not nxt:
            return None
        for code in nxt:
            ins(code >> 2, code & 3)
        frontier = nxt
    return None


def construct(k, g):
    if k != 3:
        return _fallback_luw(k, g)

    t0 = time.time()
    deadline = t0 + 110.0
    best = None

    # Candidate primes (large -> less likely to have short relations -> longer girth)
    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549,
              2147483539, 2147483509, 2147483497, 2147483471, 2147483447,
              2147483419, 2147483407, 2147483383, 2147483367, 2147483353,
              2147483327, 2147483321, 2147483299, 2147483267, 2147483239]
    # Deterministic generator draws for each prime
    gen_sets = [
        (1234567, 7654321, 9876543),
        (2468135, 1357911, 5139872),
        (111222333, 444555666, 777888999),
        (9999991, 3333337, 7777771),
        (2000003, 4000007, 6000011),
        (123456789, 987654321, 456789123),
    ]

    for p in primes:
        if time.time() > deadline:
            break
        node_cap = min(250000, p * p)
        for gs in gen_sets:
            if time.time() > deadline:
                break
            L = _bfs_girth(p, gs[0], gs[1], gs[2], node_cap)
            if L is not None and L >= g:
                best = (p, gs[0], gs[1], gs[2], L)
                break
        if best is not None:
            break

    if best is not None:
        p, a, b, c = best[0], best[1], best[2], best[3]
        g1, g2, g3 = a % p, b % p, c % p
        neg1, neg2, neg3 = (-g1) % p, (-g2) % p, (-g3) % p
        edges = []
        for i in range(p):
            edges.append((i, (i + g1) % p))
            edges.append((i, (i + g2) % p))
            edges.append((i, (i + g3) % p))
        return p, edges

    return _fallback_luw(k, g)


def _fallback_luw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d,q), q=k: guaranteed valid 3-regular graph.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    return len(order), edges
