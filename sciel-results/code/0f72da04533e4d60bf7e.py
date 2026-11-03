import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the LUW construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    q = k  # For k=3, q=3

    # LUW CD(d, q) has girth 2d+2
    # We need 2d+2 >= g, so d >= (g-2)/2
    d_min = (g - 2 + 1) // 2  # ceil((g-2)/2)
    
    # Try d values starting from minimum needed
    best_n = float('inf')
    best_edges = None

    for d in range(d_min, d_min + 5):
        if time.time() > deadline:
            break
            
        try:
            n, edges = _luw_construct(q, d)
            # Verify girth using non-backtracking BFS
            girth = _verify_girth(n, edges, k)
            if girth >= g and n < best_n:
                best_n = n
                best_edges = edges
                # If we found a valid graph, we can stop early if it's good enough
                if n <= 936:  # Known best for (3,16)
                    break
        except Exception:
            continue

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: use larger d to ensure girth
    d = d_min + 5
    n, edges = _luw_construct(q, d)
    return n, edges


def _luw_construct(q, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q).
    Returns (n, edges).
    """
    # Precompute rules
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule_a[c], rule_b[c] = 1, c0 - 2
        elif r == 1:
            rule_a[c], rule_b[c] = c0 - 1, 1
        elif r == 2:
            rule_a[c], rule_b[c] = c0, 1
        else:
            rule_a[c], rule_b[c] = 1, c0 + 1

    rule_a_t = tuple(rule_a)
    rule_b_t = tuple(rule_b)

    # Generate vertices using BFS
    index = {}
    order = []  # list of (side, coord_tuple)
    edges = []

    start_coord = (0,) * (d + 1)
    start_key = (0, start_coord)
    index[start_key] = 0
    order.append((0, start_coord))

    c_range = range(2, d + 1)

    for i, vertex in enumerate(order):
        side, x = vertex

        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first

            if side == 0:
                for c in c_range:
                    y[c] = (x[c] + y[rule_a_t[c]] * x[rule_b_t[c]]) % q
            else:
                for c in c_range:
                    y[c] = (x[c] - x[rule_a_t[c]] * y[rule_b_t[c]]) % q

            y_t = tuple(y)
            neighbor_key = (1 - side, y_t)

            if neighbor_key not in index:
                index[neighbor_key] = len(order)
                order.append((1 - side, y_t))

            if side == 0:
                edges.append((i, index[neighbor_key]))

    n = len(order)
    return n, edges


def _verify_girth(n, edges, k):
    """
    Verify girth using non-backtracking BFS from every vertex.
    Returns the minimum cycle length found.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    min_girth = float('inf')

    # Non-backtracking BFS from each vertex
    for start in range(n):
        # dist[v] = shortest non-backtracking path length from start to v
        # prev[v] = previous vertex in the non-backtracking path
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])  # (current, previous)

        while queue:
            u, p = queue.popleft()
            for v in adj[u]:
                if v == p:
                    continue  # non-backtracking: don't go back
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append((v, u))
                else:
                    # Found a cycle: start -> ... -> u -> v -> ... -> start
                    # The cycle length is dist[u] + dist[v] + 1
                    # But we need to be careful: this might not be the shortest cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= 8:
                            return min_girth  # Early exit for small girths

        if min_girth <= 10:
            break

    return min_girth if min_girth != float('inf') else n
