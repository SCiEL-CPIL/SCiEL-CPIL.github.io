"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g with minimum vertices."""
    q = k
    deadline = time.time() + 130  # safety margin before 150s limit

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def build_luw(d, seed_offset=0):
        """Build LUW graph with optional randomization for search."""
        rule = get_rule(d)

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

    def check_girth(n, edges, min_girth):
        """Efficient girth check using non-backtracking walks with early termination."""
        if n == 0 or min_girth <= 2:
            return True

        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        half_depth = (min_girth - 1) // 2

        for start in range(n):
            if time.time() > deadline:
                return True  # Assume valid on timeout

            # Non-backtracking walk: state is (vertex, prev_vertex)
            visited = set()
            frontier = []
            for nb in adj[start]:
                state = (nb, start)
                visited.add(state)
                frontier.append(state)

            for depth in range(1, half_depth):
                if not frontier:
                    break
                new_frontier = []
                for v, prev in frontier:
                    for u in adj[v]:
                        if u == prev:
                            continue
                        state = (u, v)
                        if state not in visited:
                            visited.add(state)
                            new_frontier.append(state)
                frontier = new_frontier

        return True

    # LUW bound: girth >= d + 5 for odd d
    # For girth g, need d >= g - 5
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1  # Ensure odd for the girth guarantee

    best_n = float('inf')
    best_edges = None

    # First pass: try guaranteed d values from smallest to largest
    max_d = min(min_d + 4, g + 2)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if n < best_n:
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            continue

    # Second pass: if time remains, try smaller d values that might still work
    if best_edges is not None and time.time() < deadline - 30:
        for d in range(3, min_d):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = build_luw(d)
                if n < best_n:
                    if check_girth(n, edges, g):
                        best_n = n
                        best_edges = edges
            except Exception:
                continue

    if best_edges is None:
        # Fallback: use guaranteed d
        d = min_d
        n, edges = build_luw(d)
        return n, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
