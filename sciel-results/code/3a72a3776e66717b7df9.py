"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0  # safety margin before 120s limit

    # For k=3, g=16, the best known cage is 936 vertices (Exoo et al. 2026).
    # The LUW construction is far from optimal. We try a few strategies:
    # 1) Try to find a smaller 3-regular graph with girth >= g via a randomized
    #    algebraic construction (cayley-like on a finite abelian group).
    # 2) Fall back to the LUW construction if no better graph is found in time.

    if k != 3:
        # For other k, just use LUW as before
        return _luw_construct(k, g)

    # For k=3, g=16: try to find a graph with fewer vertices than LUW.
    # LUW for k=3, g=16: d = g-5 = 11 (odd), so girth >= 11+5=16.
    # Number of vertices ~ q^d = 3^11 = 177147. That's way too many.
    # Actually, the LUW graph is bipartite with q^d vertices on each side,
    # so total ~ 2*q^d. For q=3, d=11: 2*177147 = 354294 vertices.
    # But the connected component of the zero point might be smaller.
    # Regardless, it's likely much larger than 936.

    # Strategy: Try a randomized search for a 3-regular graph with girth >= 16
    # on a moderate number of vertices. We'll use a simulated annealing-like
    # approach on a finite abelian group.

    # Moore bound for k=3, g=16: 2 * sum_{i=0}^{7} 3^i = 2*(1+3+9+27+81+243+729+2187) = 2*3280 = 6560?
    # Wait, the Moore bound for cubic graphs: n >= 2 * (3^(g/2) - 1) / (3-1) for even g.
    # For g=16: n >= 2 * (3^8 - 1)/2 = 3^8 - 1 = 6561 - 1 = 6560.
    # But the problem says the best lower bound is 512. Let me re-check.
    # Actually, the Moore bound is a lower bound, and 512 is likely from a different bound.
    # The problem states: "the best lower bound is 512". So we aim for n >= 512.

    # Let's try to construct a graph on n vertices for n in a reasonable range.
    # We'll use a Cayley graph on Z_n x Z_m or similar.

    # Actually, let's try a different approach: use the known 936-vertex graph if we can
    # reconstruct it, or try to find one via search.

    # Given time constraints, let's try a few small constructions:
    # 1) Generalized Petersen graphs GP(n, k) - but these are 3-regular and have specific girths.
    # 2) Random 3-regular graphs and check girth.

    # For simplicity and reliability, let's use the LUW construction but try to
    # optimize the parameters. Actually, LUW is not going to give 936 vertices.

    # Let's try a randomized approach: generate random 3-regular graphs on n vertices
    # and check if they have girth >= 16. Start with n = 936 (known to work) and
    # try smaller n.

    # But checking girth >= 16 requires non-backtracking walks, which is expensive.
    # Let's try n from 512 to 936 in steps, and for each n, try a few random graphs.

    best_n = None
    best_edges = None

    # Try n = 936 first (known to work)
    # Actually, we don't have the specific 936-vertex graph. Let's try to find one.

    # Given the complexity, let's fall back to LUW but with a time check.
    # The LUW graph for k=3, g=16 will be very large, but it's valid.

    # Actually, let's try a different construction: the (3,16)-cage is known to have
    # 936 vertices. Without the specific graph, we can't use it.

    # Let's try a randomized construction:
    # Generate a random 3-regular graph on n vertices, check girth.

    import random
    random.seed(42)

    def _check_girth(n, edges, min_girth):
        """Check if the graph has girth >= min_girth using BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Check for cycles of length < min_girth
        # Use BFS from each vertex, tracking parent to avoid backtracking
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for idx in range(len(queue)):
                curr = queue[idx]
                if dist[curr] >= min_girth // 2:
                    continue
                for neighbor in adj[curr]:
                    if neighbor == parent[curr]:
                        continue
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    else:
                        # Found a cycle
                        cycle_len = dist[curr] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Try to find a 3-regular graph with girth >= 16 on n vertices
    # Start with n = 936 (known to work) and try to go lower

    # Given time constraints, let's just use LUW as a fallback.
    # But first, try a few random graphs on n = 936.

    n_target = 936
    if _time_left() > 30:
        # Try to find a valid graph on n_target vertices
        for attempt in range(20):
            if _time_left() < 10:
                break
            # Generate a random 3-regular graph
            try:
                edges = _random_3regular(n_target)
                if edges is not None:
                    if _check_girth(n_target, edges, g):
                        best_n = n_target
                        best_edges = edges
                        break
            except:
                continue

    if best_n is not None:
        return best_n, best_edges

    # Fall back to LUW
    return _luw_construct(k, g)


def _random_3regular(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
    # Use configuration model
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # loop, reject
        if (min(u,v), max(u,v)) in edges:
            return None  # multi-edge, reject
        edges.add((min(u,v), max(u,v)))
    if len(edges) != n * 3 // 2:
        return None
    return list(edges)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
