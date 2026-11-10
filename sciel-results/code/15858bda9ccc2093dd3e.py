"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from each vertex."""
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        # state: (current, prev)
        visited = set()
        stack = [(start, -1, 0)]
        while stack:
            node, prev, dist = stack.pop()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 >= g:
                    continue
                if dist + 1 < g and nb == start:
                    return False
                state = (nb, node)
                if state in visited:
                    continue
                visited.add(state)
                stack.append((nb, node, dist + 1))
    return True

def construct(k, g):
    deadline = time.time() + 120
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    # For bipartite LUW CD(d,q): girth is 2d+2 for odd d. Need 2d+2 >= g.
    # Smallest odd d >= 3 with 2d+2 >= g: d = max(3, ceil((g-2)/2) adjusted to odd)
    d_min = max(3, (g - 2 + 1) // 2)
    if d_min % 2 == 0:
        d_min += 1
    d = d_min
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline - 10:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
