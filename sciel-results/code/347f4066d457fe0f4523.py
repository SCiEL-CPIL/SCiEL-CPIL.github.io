import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Moore bound: n_min = 2 * (1 + sum_{i=0}^{(g-3)/2} (k-1)^i) for g even
    if g % 2 == 0:
        # girth 10: cycles of length < 10 forbidden
        # Moore bound for 4-regular, girth 10: 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242
        # Actually the Moore bound formula for girth g (even): 2 * sum_{i=0}^{g/2 - 2} (k-1)^i + 2*(k-1)^{g/2-1}
        # For g=10, k=4: 2*(1 + 3 + 9 + 27) + 2*81 = 2*40 + 162 = 80 + 162 = 242
        moore = 2
        for i in range(1, g // 2 - 1):
            moore += 2 * (k - 1) ** i
    else:
        moore = 2
        for i in range(1, (g - 1) // 2):
            moore += 2 * (k - 1) ** i

    # Try known constructions
    # For (4, 10): LUW with d=7 gives n=400, d=8 gives n=500, etc.
    # The 320-vertex cage is from a different construction.
    # Let's try LUW with various d values and also try a different approach.

    best_n, best_edges = None, None

    # Try LUW with d from 5 to 15
    for d in range(5, 16):
        if time.time() > deadline - 5:
            break
        n, edges = _luw_construct(k, g, d)
        if n is None:
            continue
        if _verify_girth(n, edges, g):
            if best_n is None or n < best_n:
                best_n, best_edges = n, edges
            break

    if best_n is not None:
        return best_n, best_edges

    # Fallback
    d_default = max(5, g - 3)
    return _luw_construct(k, g, d_default)


def _verify_girth(n, edges, g):
    """Verify that the graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] >= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True


def _luw_construct(k, g, d):
    """Lazebnik-Ustimenko-Woldar construction for k-regular graphs of large girth."""
    q = k
    if q == 4:
        # Use GF(4) = {0,1,2,3} with addition XOR, multiplication via table
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for each column c (2 to d)
    # The LUW rule: for column c, we have coefficients a_c, b_c
    # y[c] = x[c] + y[a_c] * x[b_c] (side 0)
    # y[c] = x[c] - x[a_c] * y[b_c] (side 1)
    rule = {}
    for c in range(2, d + 1):
        # Standard LUW: a_c = c-1, b_c = 1, but we need to ensure no short cycles
        # The original LUW uses a specific pattern
        # For girth g, we need d >= g/2 - 1 approximately
        # Use the pattern from the original paper
        rule[c] = (c - 1, 1)

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
    return len(order), edges
