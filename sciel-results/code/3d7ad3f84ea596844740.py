# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0
    best_n = None
    best_edges = None

    # --- LUW Construction (fallback) ---
    def luw_construct(k, g):
        q = k
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
        add_table = [[add(i, j) for j in range(q)] for i in range(q)]
        sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > deadline - 20:
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # --- Todd-Coxeter / Coset Enumeration Approach ---
    # For a 4-regular graph of girth 10, we can model it as a Schreier graph
    # of a group with 2 generators (a, b) where each vertex has edges labeled
    # a, a^{-1}, b, b^{-1}. The girth constraint means no reduced word of
    # length < 10 evaluates to the identity.
    #
    # We use a random "peeling" / random walk approach to build a graph
    # incrementally, ensuring no short cycles.

    def try_coset_enumeration(k, g, time_budget):
        """
        Attempt to construct a k-regular graph of girth g using a
        randomized coset enumeration / random graph construction.
        Returns (n, edges) or None.
        """
        if time_budget < 5:
            return None

        # Moore bound for reference
        if g % 2 == 0:
            r = g // 2
            moore = 2 * sum((k - 1) ** i for i in range(r))
        else:
            moore = k * sum((k - 1) ** i for i in range((g - 1) // 2))

        # We'll try to build a graph with n vertices starting from a reasonable size
        # and use random construction with girth checking.

        # Strategy: build a random 4-regular graph and check girth,
        # or use a structured approach.

        # For (4,10), known smallest is 320. Try sizes from 300 down.
        # But constructing and verifying is expensive. Instead, use a
        # structured random approach.

        # Use a "random lift" approach: start with a base graph and lift it.
        # Or: use the fact that (4,10)-cage can be constructed via
        # voltage graphs over a base.

        # Simpler: try random 4-regular graphs of decreasing size and check girth.
        # This is very expensive, so limit attempts.

        # Alternative: use a greedy construction.
        # Start with n vertices, assign 4 neighbors to each, ensuring
        # no cycles shorter than g.

        # For now, implement a basic random search with limited budget.
        # If time permits, do a few trials.

        def generate_random_4reg(n):
            """Generate a random 4-regular graph on n vertices."""
            if n < 4:
                return None
            # Use configuration model
            half_edges = [i for i in range(n) for _ in range(2)]  # 2 pairs per vertex for degree 4
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            edge_set = set()
            for i in range(0, 2 * n, 2):
                u, v = half_edges[i], half_edges[i + 1]
                if u == v:
                    return None  # self-loop
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    return None  # multi-edge
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)

            # Check degree
            for i in range(n):
                if len(adj[i]) != 4:
                    return None
            return adj

        def check_girth(adj, n, g):
            """Check if graph has girth at least g using BFS from each vertex."""
            for start in range(n):
                # BFS with parent tracking to avoid backtracking
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif dist[v] <= dist[u] + 1:
                            # Found a cycle of length dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True

        # Try a few random constructions
        attempts = 0
        max_attempts = 50  # limit to keep within time
        for n_try in range(max(best_n - 20, moore), max(best_n - 20, moore) + 1):
            if time.time() > deadline - 10:
                break
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                adj = generate_random_4reg(n_try)
                if adj is None:
                    continue
                if check_girth(adj, n_try, g):
                    edges = []
                    for u in range(n_try):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_try, edges
        return None

    # Try the coset enumeration / random approach if we have time
    remaining = deadline - time.time() - 5
    if remaining > 10:
        result = try_coset_enumeration(k, g, remaining)
        if result:
            n_try, edges_try = result
            if n_try < best_n:
                best_n = n_try
                best_edges = edges_try

    return best_n, best_edges
# EVOLVE-BLOCK-END
