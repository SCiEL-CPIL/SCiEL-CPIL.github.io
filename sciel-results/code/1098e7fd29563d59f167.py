import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 105.0

def _deadline():
    return time.time() >= _DEADLINE

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    if q == 4:
        add_t = np.array([0, 0, 0, 0], dtype=np.int8)
        add_t2 = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int8)
        add_op = add_t2
        mul_t = np.zeros((4, 4), dtype=np.int8)
        mul_t[1] = [0, 1, 2, 3]
        mul_t[2] = [0, 2, 3, 1]
        mul_t[3] = [0, 3, 1, 2]
    else:
        add_op = lambda a, b: (a + b) % q
        mul_t = np.zeros((q, q), dtype=np.int64)
        for i in range(q):
            for j in range(q):
                mul_t[i, j] = (i * j) % q

    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    # Build adjacency using BFS from start
    # Vertex represented as (side, x_1, ..., x_d) where x is in [0, q)^d
    # side 0: x[0] is dummy (None), so really x[1..d]
    # side 1: x[1] is the "first" variable

    # Number of vertices: q^d * 2 (side 0 has q^d, side 1 has q^d)
    # Actually side 0: (0, (None, x1, ..., xd)) - x1..xd free -> q^d vertices
    # side 1: (1, (y1, y2, ..., yd)) - all free -> q^d vertices

    total = 2 * q ** d
    if total > 2000000:
        return None, None, None

    # Map vertex to index
    def vid(side, x):
        # x is tuple of d values (for side 0, x[0] is ignored/dummy)
        idx = x[0]
        for c in range(1, d + 1):
            idx = idx * q + x[c]
        return side * (q ** d) + idx

    # Precompute neighbours
    adj = [set() for _ in range(total)]

    # For side 0 vertices: (0, (None, x1, ..., xd))
    # Neighbour: for each first in range(q):
    #   y[1] = first
    #   for c in 2..d: y[c] = x[c] + y[a] * x[b]  (mod q)
    #   neighbour is (1, y)

    for side0_idx in range(q ** d):
        # decode x
        x = [0] * (d + 1)
        tmp = side0_idx
        for c in range(d, 0, -1):
            x[c] = tmp % q
            tmp //= q
        # x[0] = 0 (dummy)

        v0 = vid(0, tuple(x))

        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rc[c]
                ya = y[a] if a >= 1 else 0
                xb = x[b]
                if q == 4:
                    y[c] = (x[c] + mul_t[ya, xb]) % 4
                else:
                    y[c] = (x[c] + ya * xb) % q
            v1 = vid(1, tuple(y))
            adj[v0].add(v1)
            adj[v1].add(v0)

    # For side 1 vertices: (1, (y1, ..., yd))
    # Neighbour: for each first in range(q):
    #   x[1] = first
    #   for c in 2..d: x[c] = y[c] - y[a] * x[b]  (mod q)
    #   Wait, let me re-read the rule.
    #   The rule for side 0: y[c] = x[c] + y[a]*x[b]
    #   The rule for side 1: x[c] = y[c] - x[a]*y[b]  ... hmm, let me re-check.
    #
    # Actually looking at the original code:
    #   side 0: y[c] = add(x[c], mul(y[a], x[b]))
    #   side 1: y[c] = sub(x[c], mul(x[a], y[b]))
    #
    # Wait, the original code uses "y" for the target and "x" for the source.
    # Let me re-read:
    #   def neighbours(vertex):
    #       side, x = vertex
    #       for first in range(q):
    #           y = [None, first] + [0] * (d - 1)
    #           for c in range(2, d + 1):
    #               a, b = rc[c]
    #               y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
    #           yield 1 - side, tuple(y)
    #
    # So for side 0 (x is source, y is target):
    #   y[c] = x[c] + y[a] * x[b]
    # For side 1 (x is source, y is target):
    #   y[c] = x[c] - x[a] * y[b]

    # For side 1 vertices, we need to iterate and find neighbours.
    # But actually we already added edges from side 0 to side 1.
    # The LUW graph is bipartite, so every edge connects side 0 to side 1.
    # We've already added all edges by iterating over side 0 vertices.
    # But we need to make sure each vertex has exactly q neighbours.

    # Let's verify: side 0 vertex has q neighbours (one per "first" value).
    # Side 1 vertex: for each side 0 vertex x and each first, we computed y.
    # The mapping from (x, first) -> y is a bijection? Let's assume so for now.

    # Check degrees
    for v in range(total):
        if len(adj[v]) != q:
            # This shouldn't happen for valid LUW graphs
            pass

    # Convert to edge list
    edges = []
    for i in range(total):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))

    return total, edges, adj

def _girth_check(adj, g, n):
    """Check if minimum cycle length is >= g using BFS from each directed edge."""
    if n == 0:
        return 0

    # Use non-backtracking BFS
    # For each starting edge (u, v), BFS in the directed line graph
    # A cycle of length L is found when we return to (u, v) after L steps
    # Actually, we need to find the shortest cycle.

    # Simpler: for each vertex, BFS to find shortest cycle through it
    # But that's O(n * (n + m)) which is too slow.

    # Better: for each directed edge (u, v), do BFS from (u, v) in the
    # directed edge graph. A cycle of length L corresponds to a path of
    # length L-1 from (u, v) back to (u, v) where the last step closes.
    # Actually, a cycle of length L: v0-v1-...-vL-1-v0
    # Starting from directed edge (v0, v1), after L-1 more steps we reach (vL-1, v0).
    # So we're looking for a path of length L-1 from (v0,v1) to (vL-1, v0).

    # Even simpler: just do BFS from each vertex to find the shortest cycle.
    # For vertex u, BFS from all neighbours of u. If we find u again at depth d,
    # cycle length is d+1.

    # This is O(n*(n+m)) which for n=1116, m=2790 is about 1116*3906 ≈ 4.3M, fine.

    best = float('inf')

    for start in range(n):
        # BFS from all neighbours of start
        dist = [-1] * n
        from collections import deque
        dq = deque()
        for nb in adj[start]:
            dist[nb] = 1
            dq.append(nb)

        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v == start:
                    cand = dist[u] + 1
                    if cand < best:
                        best = cand
                    if best <= g:
                        return best
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)

        if best <= g:
            return best

    return best if best < float('inf') else None

def construct(k, g):
    # k=5, g=9 for the cage problem
    # LUW construction: q-regular, girth >= d+5 for odd d
    # We need q = k = 5, girth >= g = 9
    # So d+5 >= 9 => d >= 4. For odd d: d >= 5.
    # But the LUW construction requires q to be a prime power.
    # q=5 is prime, so d=5 gives girth >= 10 >= 9. Good.
    # n = 2 * q^d = 2 * 5^5 = 6250. That's quite large.

    # Actually wait - the LUW construction gives a bipartite graph.
    # The girth bound is girth >= 2*ceil((d+1)/2) + 1 or something...
    # Let me just try different d values and check.

    candidates = []
    q = k  # q must equal k for k-regular

    # d must be odd for the LUW construction to work properly
    # girth >= d + 5 (for the bipartite LUW graph)
    # We need girth >= g, so d >= g - 5
    # d must be odd
    d_min = g - 5
    if d_min % 2 == 0:
        d_min += 1
    d_min = max(3, d_min)

    for d in range(d_min, d_min + 10, 2):
        candidates.append((d, q))

    best_n = float('inf')
    best_edges = None

    for (d, q) in candidates:
        if _deadline():
            break

        n_expected = 2 * q ** d
        if n_expected > 50000:
            continue

        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue

        if n is None:
            continue

        # Verify degree
        deg = [0] * n
        for (i, j) in edges:
            deg[i] += 1
            deg[j] += 1
        if not all(x == k for x in deg):
            continue

        # Check girth
        if _deadline():
            # Return what we have if it's regular
            if best_edges is None:
                return n, edges
            break

        gi = _girth_check(adj, g, n)
        if gi is None or gi < g:
            continue

        if n < best_n:
            best_n = n
            best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: try without girth check
    for (d, q) in candidates:
        if _deadline():
            break
        n_expected = 2 * q ** d
        if n_expected > 50000:
            continue
        try:
            n, edges, adj = _luw_graph(d, q)
            if n is None:
                continue
            deg = [0] * n
            for (i, j) in edges:
                deg[i] += 1
                deg[j] += 1
            if all(x == k for x in deg):
                return n, edges
        except Exception:
            continue

    return 0, []
