"""Rewritten program: explicit constructions (LUW + circulant) with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 150.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _luw_construct(q, d):
    # Lazebnik-Ustimenko-Woldar CD(d,q): bipartite q-regular graph, girth d+5.
    # Returns (n, edges) for the connected component of the zero point.
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _circulant_girth_ok(n, S, g):
    # S: list of offsets (1..n//2), graph on Z_n with edges +-S.
    # True iff no cycle of length < g. Fast bitset BFS per start, early stop.
    half = n // 2
    base = [1 << s for s in S]
    for start in range(n):
        # bitset of visited nodes
        seen = 1 << start
        frontier = seen
        for _ in range(g - 1):
            nxt = 0
            for s in S:
                nxt |= ((frontier << s) | (frontier >> s)) & ((1 << n) - 1)
            nxt &= ~seen
            if nxt:
                return False
            seen |= nxt
            frontier = nxt
    return True


def _find_circulant(k, g, max_tries):
    # Search for a circulant k-regular graph of girth >= g with as few vertices as possible.
    rng = random.Random(12345)
    best = None
    # Moore-ish lower bound to bound the search window
    if g % 2 == 0:
        g2 = g // 2
        lower = 1 + k * (k ** (g2 - 1) - 1) // (k - 1)
    else:
        g2 = (g - 1) // 2
        lower = 1 + k * (k ** g2 - 1) // (k - 1)
    for n in range(max(k + 1, lower), lower + 60):
        if _time_left() < 3.0:
            break
        if (k * n) % 2 != 0:
            continue
        if k == 2:
            continue
        for _ in range(max_tries):
            if _time_left() < 3.0:
                break
            S = rng.sample(range(1, n // 2 + 1), k // 2)
            if _circulant_girth_ok(n, S, g):
                edges = []
                for i in range(n):
                    for s in S:
                        edges.append((i, (i + s) % n))
                if best is None or n < best[0]:
                    best = (n, edges)
                    break  # found for this n; try smaller? n increasing, so first n is smallest
        if best is not None:
            break
    return best


def construct(k, g):
    results = []

    # 1) LUW family (works for odd girth primarily; q = k)
    try:
        q = k
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 1:
            d -= 1
        if d >= 3:
            n, edges = _luw_construct(q, d)
            results.append((n, edges))
    except Exception:
        pass

    # 2) Circulant search (covers even girth, e.g. (5,10), (5,12))
    try:
        c = _find_circulant(k, g, max_tries=4000)
        if c is not None:
            results.append(c)
    except Exception:
        pass

    if not results:
        # Fallback: LUW with a safe default
        n, edges = _luw_construct(k, max(3, g - 5))
        return n, edges

    # Return the one with the fewest vertices
    results.sort(key=lambda x: x[0])
    return results[0]
# EVOLVE-BLOCK-END
