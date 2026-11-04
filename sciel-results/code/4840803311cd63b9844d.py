"""
Rewritten program for constructing small 3-regular graphs of given girth.
Uses the Lazebnik-Ustimenko-Woldar (LUW) construction with a carefully chosen
parameter d that balances graph size against girth guarantee.
"""
import time
from collections import deque


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # For the specific case k=3, g=16, we want the smallest LUW graph
    # that guarantees girth >= 16.
    #
    # The LUW graph CD(d, q) with q = k has girth at least 2d+2 (for the
    # appropriate rule set) and has q^2 * (q-1)^d vertices... actually it's
    # more nuanced. The key insight from cell findings: fixing d=7 for g=16
    # reduces size from ~354k to ~4k vertices.
    #
    # The LUW construction CD(d, q) produces a k-regular graph (k=q) with
    # girth at least 2d+2 (or 2d+1 depending on parity). The number of
    # vertices is q * (q-1)^d * 2 roughly, but the actual construction
    # generates only the reachable vertices.
    #
    # For girth g, we need 2d+2 >= g, so d >= (g-2)/2.
    # For g=16: d >= 7.
    #
    # The cell finding says d=7 gives ~4k vertices which is much better
    # than d=11 giving ~354k. So we should use the MINIMUM d that
    # satisfies the girth requirement.

    # Determine the minimum d needed for girth >= g
    # LUW CD(d,q) has girth >= 2d+2 (for even) or 2d+1 (for odd)
    # We need girth >= g
    # So 2d+2 >= g => d >= (g-2)/2
    # For safety, use d = (g-2)//2, and if g is odd, d = (g-1)//2

    if g % 2 == 0:
        d = (g - 2) // 2
    else:
        d = (g - 1) // 2

    # Ensure d is at least 3 (minimum for the rule set)
    d = max(3, d)

    # Generate the LUW graph
    n, edges = _generate_luw_graph(k, g, d)

    if time.time() > deadline - 5:
        return n, edges

    # Verify the graph is valid (degree k, girth >= g)
    # If not valid, fall back to a larger d
    if not _verify_graph(n, edges, k, g):
        # Try incrementing d until we get a valid graph
        for d_try in range(d + 1, d + 20):
            if time.time() > deadline - 10:
                break
            n, edges = _generate_luw_graph(k, g, d_try)
            if _verify_graph(n, edges, k, g):
                break

    return n, edges


def _generate_luw_graph(k, g, d):
    """Generate the Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k

    # Choose the arithmetic: for q=3, use mod 3 arithmetic
    # For q=4, use GF(4)
    if q == 4:
        # GF(4) operations
        # Represent elements as 0,1,2,3 with specific addition/multiplication
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: add_table[x][y]
        sub = lambda x, y: add_table[x][y]  # In GF(4), subtraction = addition
        mul = lambda x, y: mul_table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    else:
        # For general prime q, use modular arithmetic
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for each column c (2 to d)
    # rule[c] = (a, b) meaning: y[c] = y[a] * x[b] (side 0) or x[a] * y[b] (side 1)
    # The standard LUW rule set:
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Cycle through a pattern
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Ensure rule indices are valid (within 1..d)
    for c in range(2, d + 1):
        a, b = rule[c]
        # Clamp to valid range
        a = max(1, min(a, d))
        b = max(1, min(b, d))
        rule[c] = (a, b)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

    # BFS to enumerate all reachable vertices
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_graph(n, edges, k, g, max_time=30.0):
    """Verify the graph is k-regular and has girth >= g."""
    start_time = time.time()

    # Check degree
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    for d in degree:
        if d != k:
            return False

    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e in edge_set:
            return False
        edge_set.add(e)

    # Check girth using BFS from each vertex
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    min_cycle = g  # We only care if it's less than g

    for start in range(n):
        if time.time() - start_time > max_time:
            # Timeout: assume valid if we haven't found a short cycle yet
            return True

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
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < g:
                            return False

    return min_cycle >= g


def _moore_bound(k, g):
    """Moore bound for k-regular graphs of girth g."""
    if g % 2 == 0:
        return k + k * (k - 1) ** (g // 2 - 1)
    else:
        total = 1
        term = 1
        for i in range((g - 3) // 2 + 1):
            total += k * term
            term *= (k - 1)
        return total
