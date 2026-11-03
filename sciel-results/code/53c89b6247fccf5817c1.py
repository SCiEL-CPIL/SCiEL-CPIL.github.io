import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin before 120s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    def seed_construct(k, g):
        """LUW construction as fallback - returns n=2560 for k=5,g=9"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
            rc = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]

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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def has_short_cycle(adj, n, g):
        """Check if graph has any cycle of length < g using BFS from each vertex."""
        max_depth = (g - 1) // 2  # For odd girth g, we search up to depth (g-1)/2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle: dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False

    def generate_random_k_regular(n, k, rng):
        """Generate a random k-regular simple graph using configuration model with repair."""
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)

        # Try multiple times to get a simple graph
        for attempt in range(100):
            rng.shuffle(stubs)
            edges = set()
            valid = True
            adj = [[] for _ in range(n)]

            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
                adj[u].append(v)
                adj[v].append(u)

            if valid:
                return list(edges), adj

        return None, None

    def try_n(n, k, g, rng, time_budget):
        """Try to find a valid k-regular graph with girth >= g on n vertices."""
        start = time.time()
        max_attempts = 200

        for attempt in range(max_attempts):
            if time.time() - start > time_budget:
                return None
            if time.time() > deadline:
                return None

            edges, adj = generate_random_k_regular(n, k, rng)
            if edges is None:
                continue

            # Check girth
            if not has_short_cycle(adj, n, g):
                return edges

        return None

    # Strategy: Try random search for n from moore_bound upward
    # Start with a reasonable n and increase if needed
    best_n, best_edges = None, None

    # First, try a range of n values
    # For k=5, g=9: moore_bound = 426
    # We'll try n from 426 to maybe 600, with more attempts for smaller n

    rng = random.Random(42)

    # Try several n values with limited time each
    n_values = list(range(moore_bound, moore_bound + 200))

    # Shuffle to try different n values
    # But we want to return the smallest valid n, so try in order
    # However, small n is harder, so let's try a few small ones first with more time

    # Actually, let's try from a moderate n downward to find the smallest
    # But we need to be systematic. Let's try n = 426, 430, 440, 450, 460, 480, 500
    # with decreasing time budgets

    candidate_ns = [moore_bound, moore_bound + 10, moore_bound + 20,
                    moore_bound + 50, moore_bound + 100, moore_bound + 150]

    time_per_n = 8.0  # seconds per n value

    for n in candidate_ns:
        if time.time() > deadline - 2.0:
            break

        edges = try_n(n, k, g, rng, time_per_n)
        if edges is not None:
            best_n = n
            best_edges = edges
            break  # Found smallest valid n in our candidates

    # If we found a valid graph, return it
    if best_n is not None:
        return best_n, best_edges

    # Fallback: Try a larger n with more attempts
    # Or just return the seed construction
    if time.time() > deadline - 1.0:
        return seed_construct(k, g)

    # Try n = 800 with more attempts
    n = 800
    edges = try_n(n, k, g, rng, 10.0)
    if edges is not None:
        return n, edges

    # Final fallback
    return seed_construct(k, g)
