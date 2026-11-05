import time
import random
import math
import numpy as np
from collections import deque, defaultdict

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For k=4, g=10, attempt to find a smaller cage via voltage graph / automaton approach
    if k == 4 and g == 10:
        deadline = time.time() + 110

        # Moore bound for (4,10)-cage
        # n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(floor(g/2)-1))
        # For even girth g, n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2 - 1))
        # For g=10, g/2=5, so n >= 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484? No, that's wrong.
        # Actually the Moore bound for even girth g=2r is: n >= k * sum_{i=0}^{r-1} (k-1)^i
        # For k=4, r=5: n >= 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484. That seems too high.
        # Let me recalculate. The Moore bound for a k-regular graph with girth g:
        # If g is even, g = 2r, then n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
        # For k=4, g=10, r=5: n >= 4 * (1 + 3 + 9 + 27 + 81) = 484. But the problem says the best lower bound is 243.
        # I think I'm confusing the Moore bound. Let me look it up properly.
        # The Moore bound for k-regular graphs with girth g:
        # If g is odd, g = 2r+1, then n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # If g is even, g = 2r, then n >= k * sum_{i=0}^{r-1} (k-1)^i
        # For k=4, g=10=2*5, r=5: n >= 4 * (1 + 3 + 9 + 27 + 81) = 484.
        # But the task says the best lower bound is 243 and the smallest known is 320. So maybe the Moore bound isn't tight here.
        # Actually, I think I need to just try to find a graph with n < 320.

        # Voltage graph approach:
        # Start with a small base 4-regular graph (like K4,4 or a smaller cage) and assign voltages
        # to edges from a group G. The lift will be |G| * |V_base| vertices.
        # We want the lift to have girth >= 10.

        # Let's try a different approach: use a known construction or search for a smaller cage.
        # The Hoffman-Singleton graph is a (7,5)-cage with 50 vertices, not (4,10).
        # For (4,10), the known smallest is 320 vertices.

        # Let's try to construct a (4,10)-cage with fewer than 320 vertices using a
        # voltage graph approach with a small base graph.

        # Base graph: K4 (4 vertices, 4-regular, girth 4)
        # Voltages from Z_m for some m. The lift will have 4*m vertices.
        # We need to find voltages such that the lift has girth >= 10.

        def try_voltage_graph(base_edges, m, time_budget):
            """
            Try to find voltage assignments to base_edges such that the lift has girth >= 10.
            base_edges: list of (u, v) edges in the base graph
            m: the voltage group size (voltages are in Z_m)
            Returns (n, edges) if successful, None otherwise.
            """
            n_base = max(u for u, v in base_edges) + 1
            n = n_base * m
            deadline = time.time() + time_budget

            # Initialize voltages randomly
            voltages = {}
            for e in base_edges:
                voltages[e] = random.randint(0, m - 1)
                voltages[(e[1], e[0])] = (-voltages[e]) % m

            best_girth = 0
            best_voltages = dict(voltages)

            for iteration in range(10000):
                if time.time() > deadline:
                    break

                # Compute the girth of the lift
                girth = compute_lift_girth(base_edges, voltages, n_base, m)
                if girth >= 10:
                    # Found a valid voltage assignment
                    edges = []
                    for u in range(n_base):
                        for v in range(n_base):
                            if (u, v) in voltages:
                                for i in range(m):
                                    u_lift = u * m + i
                                    v_lift = v * m + (i + voltages[(u, v)]) % m
                                    if u_lift < v_lift:
                                        edges.append((u_lift, v_lift))
                    return n, edges

                if girth > best_girth:
                    best_girth = girth
                    best_voltages = dict(voltages)

                # Mutate a random voltage
                e = random.choice(base_edges)
                voltages[e] = (voltages[e] + random.randint(-2, 2)) % m
                voltages[(e[1], e[0])] = (-voltages[e]) % m

            return None

        def compute_lift_girth(base_edges, voltages, n_base, m):
            """
            Compute the girth of the voltage graph lift.
            Uses BFS from each vertex to find the shortest cycle.
            """
            n = n_base * m
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for (u, v) in base_edges:
                vol = voltages[(u, v)]
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + (i + vol) % m
                    adj[u_lift].append(v_lift)

            # Find the shortest cycle using BFS from each vertex
            min_girth = float('inf')
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
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
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                if min_girth <= 4:
                    break
            return min_girth if min_girth != float('inf') else 100

        # Try voltage graph with K4 base
        base_edges_k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # K4 is 3-regular, not 4-regular. We need a 4-regular base graph.
        # Let's use K5 (5 vertices, 4-regular, girth 3)
        base_edges_k5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]

        # Try different m values
        for m in [80, 64, 50, 40, 32, 25, 20, 16]:
            if time.time() > deadline - 10:
                break
            result = try_voltage_graph(base_edges_k5, m, 5)
            if result:
                return result

        # If voltage graph didn't work, try random search for a smaller cage
        def generate_random_4_regular(n):
            if n % 2 != 0:
                return None
            half_edges = [i for i in range(n) for _ in range(4)]
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    return None
                adj[u].append(v)
                adj[v].append(u)
            for u in range(n):
                if len(adj[u]) != 4 or len(set(adj[u])) != 4:
                    return None
            return adj

        def check_girth(adj, n, max_girth=9):
            """
            Returns True if girth > max_girth.
            """
            A = np.zeros((n, n), dtype=int)
            for u in range(n):
                for v in adj[u]:
                    A[u, v] = 1
            for L in range(3, max_girth + 1):
                A = A @ A
                if np.any(np.diag(A) > 0):
                    return False
            return True

        # Try random search for n < 320
        for n_target in range(244, 320, 2):
            if time.time() > deadline - 5:
                break
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                adj = generate_random_4_regular(n_target)
                if not adj:
                    continue
                if check_girth(adj, n_target, 9):
                    edges = []
                    for u in range(n_target):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_target, edges

        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
