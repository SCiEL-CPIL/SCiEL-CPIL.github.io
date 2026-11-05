# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- LUW Construction ---
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
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
            if time.time() > deadline - 10:
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

    # --- Moore Bound ---
    if g % 2 == 0:
        m = g // 2
        moore = k * sum((k-1)**i for i in range(m))
    else:
        m = (g - 1) // 2
        moore = k * sum((k-1)**i for i in range(m))

    # --- Local Search: try to find a smaller graph ---
    if n_luw > moore * 1.1:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            ls_result = local_search_construct(k, g, moore, n_luw, remaining)
            if ls_result:
                n_ls, edges_ls = ls_result
                if n_ls < n_luw:
                    return n_ls, edges_ls

    return n_luw, edges_luw

def local_search_construct(k, g, n_min, n_max, time_budget):
    """
    Attempt to construct a smaller k-regular graph with girth >= g
    using a randomized approach with girth verification.
    """
    # For k=4, g=10: try to find a graph with n < n_luw
    # Strategy: start with a random k-regular graph and check girth
    # If girth is too small, try to repair by swapping edges

    # Use a simple random regular graph generation + girth check
    # This is a heuristic; may not find anything quickly

    import random as _rand
    from collections import deque as _deque

    def check_girth(n, edges, target_girth):
        """Check if graph has girth >= target_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for start in range(n):
            # BFS with parent tracking to find shortest cycle
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = _deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def is_k_regular(n, edges, k):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return all(d == k for d in deg)

    def is_simple(edges, n):
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True

    # Try a few random constructions
    best_n = n_max
    best_edges = None

    # Attempt: random regular graph via configuration model
    for attempt in range(20):
        if time.time() > time_budget + 5:
            break

        # Try n = n_luw - 1, n_luw - 2, etc., down to moore
        for n_try in range(min(n_max - 1, n_max - 5), max(moore, n_max - 20), -1):
            if n_try < moore:
                break
            if n_try * k % 2 != 0:
                continue

            # Configuration model
            half_edges = list(range(n_try * k // 2))
            vertices = []
            for i in range(n_try):
                vertices.extend([i] * k)

            _rand.shuffle(vertices)

            # Build edges
            edges_try = []
            valid = True
            for i in range(n_try * k // 2):
                u, v = vertices[2*i], vertices[2*i+1]
                if u == v:
                    valid = False
                    break
                edges_try.append((u, v))

            if not valid:
                continue

            if not is_simple(edges_try, n_try):
                continue

            if not is_k_regular(n_try, edges_try, k):
                continue

            # Check girth
            if check_girth(n_try, edges_try, g):
                best_n = n_try
                best_edges = edges_try
                break

        if best_n < n_max:
            break

    if best_edges is not None:
        return best_n, best_edges
    return None

# EVOLVE-BLOCK-END
