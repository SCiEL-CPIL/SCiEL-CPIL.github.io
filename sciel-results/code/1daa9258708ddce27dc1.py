import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    best_result = None
    best_n = float('inf')

    def _try_record(n, edges):
        nonlocal best_result, best_n
        if n < best_n:
            best_n = n
            best_result = (n, edges)

    # --- LUW fallback (from seed) ---
    def _luw_graph(k, g):
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

    # --- Fast girth verification (numpy BFS) ---
    def _check_girth(adj, n, g_limit):
        max_depth = (g_limit - 1) // 2
        adj_np = np.array(adj, dtype=np.int32)
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                neighbors = adj_np[u]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- Voltage Assignment / Biggs-Hoffman Construction ---
    def _voltage_graph(base_edges, voltages, N):
        """
        Construct a 3-regular graph via voltage assignment over Z_N.
        base_edges: list of (u, v) where u,v are base vertices
        voltages: dict mapping (u,v) -> voltage in Z_N (for u<v)
        Returns (n, edges) or (None, None) if not valid.
        """
        base_verts = set()
        for u, v in base_edges:
            base_verts.add(u)
            base_verts.add(v)
        base_verts = sorted(base_verts)
        nv = len(base_verts)
        n = nv * N
        if n < 2:
            return None, None

        # Map base vertex to index
        vidx = {v: i for i, v in enumerate(base_verts)}

        adj = [[] for _ in range(n)]
        edge_set = set()

        for u, v in base_edges:
            vi, vj = vidx[u], vidx[v]
            vol = voltages.get((u, v), voltages.get((v, u), 0))
            for a in range(N):
                i = vi * N + a
                b = (a + vol) % N
                j = vj * N + b
                if i == j:
                    return None, None  # loop
                if (i, j) in edge_set or (j, i) in edge_set:
                    return None, None  # parallel edge
                edge_set.add((i, j))
                adj[i].append(j)
                adj[j].append(i)

        # Check 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None

        edges = sorted(edge_set)
        return n, edges

    def _search_voltage_graphs(N_max, time_limit):
        """
        Search for small 3-regular graphs with girth >= g via voltage assignments.
        Uses various base graphs (trees, cycles) with voltages over Z_N.
        """
        results = []

        # Strategy 1: Base = path of length 2 (3 vertices, 2 edges) -> not 3-regular base
        # Need base to be 3-regular. Small 3-regular base graphs:
        # K4 (4 vertices, 6 edges)
        # K_{3,3} (6 vertices, 9 edges)
        # Octahedral (6 vertices, 12 edges) - too dense
        # Petersen (10 vertices, 15 edges)

        # K4 base
        k4_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        # K_{3,3} base
        k33_edges = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        # Petersen base
        pet_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (0,5),(1,6),(2,7),(3,8),(4,9),
                     (5,7),(7,9),(9,6),(6,8),(8,5)]

        base_graphs = [
            ("K4", k4_edges),
            ("K33", k33_edges),
            ("Petersen", pet_edges),
        ]

        for base_name, base_edges in base_graphs:
            if time.time() > time_limit:
                break
            nv = len(set(v for e in base_edges for v in e))
            # Try various N values
            for N in range(2, N_max + 1):
                if time.time() > time_limit:
                    break
                n_total = nv * N
                if n_total < 512:  # Moore bound
                    continue
                if n_total > 2000:
                    break

                # For small N, enumerate voltage assignments
                # Each edge gets a voltage in Z_N
                # Number of edges * log(N) choices
                # For K4: 6 edges, for K33: 9 edges, for Petersen: 15 edges

                # Use random sampling for larger edge counts
                n_edges = len(base_edges)
                if n_edges * N <= 200:
                    # Exhaustive for small cases
                    pass  # Too complex to enumerate all; use random
                else:
                    # Random sampling
                    n_samples = min(500, max(50, int((time_limit - time.time()) * 20)))
                    for _ in range(n_samples):
                        if time.time() > time_limit:
                            break
                        voltages = {}
                        for u, v in base_edges:
                            key = (u, v) if u < v else (v, u)
                            voltages[key] = random.randint(0, N - 1)

                        n, edges = _voltage_graph(base_edges, voltages, N)
                        if n is None:
                            continue
                        # Quick girth check
                        adj = [[] for _ in range(n)]
                        for i, j in edges:
                            adj[i].append(j)
                            adj[j].append(i)
                        if _check_girth(adj, n, g):
                            results.append((n, edges))
                            _try_record(n, edges)

        return results

    # --- Circulant construction (fast) ---
    def _circulant_3reg(n, a, b):
        """3-regular circulant: connect i to i+a, i-a, i+b, i-b mod n.
        Need a, b, -a, -b all distinct and nonzero mod n."""
        if n < 6:
            return None, None
        offsets = {a % n, (-a) % n, b % n, (-b) % n}
        if len(offsets) != 4 or 0 in offsets:
            return None, None
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i < j:
                    edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        return n, edges

    # --- Main Logic ---
    if k == 3 and g == 16:
        moore_bound = 512

        # Phase 1: Search circulants (fast, many candidates)
        # A 3-regular circulant C(n; a, b) connects i to i±a, i±b mod n
        # For girth >= 16, need n large enough
        # Try n from Moore bound upward
        for n in range(moore_bound, 1500, 2):
            if time.time() > deadline - 60:
                break
            if n % 2 != 0:
                continue
            # Try pairs (a, b) with a < b, both not 0, n/2
            # and 2a != 2b, a+b != 0, a-b != 0 mod n (ensures 4 distinct offsets)
            # For girth, we want a, b to be "spread out"
            a_max = n // 3
            for a in range(1, a_max + 1):
                if time.time() > deadline - 60:
                    break
                for b in range(a + 1, n // 2):
                    if time.time() > deadline - 60:
                        break
                    # Check 4 distinct offsets
                    if (2 * a) % n == 0 or (2 * b) % n == 0:
                        continue
                    if (a + b) % n == 0 or (a - b) % n == 0:
                        continue
                    n_c, edges_c = _circulant_3reg(n, a, b)
                    if n_c is None:
                        continue
                    adj = [[] for _ in range(n_c)]
                    for u, v in edges_c:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _check_girth(adj, n_c, g):
                        _try_record(n_c, edges_c)
                        # Found one; try to find smaller
                        break
                if best_result is not None:
                    break

        # Phase 2: Voltage assignment search (Biggs-Hoffman family)
        if time.time() < deadline - 30:
            _search_voltage_graphs(N_max=60, time_limit=deadline - 30)

        # Phase 3: LUW fallback
        if time.time() < deadline - 5:
            n_luw, edges_luw = _luw_graph(k, g)
            _try_record(n_luw, edges_luw)

    else:
        # Generic fallback
        n_luw, edges_luw = _luw_graph(k, g)
        _try_record(n_luw, edges_luw)

    if best_result is not None:
        return best_result

    # Absolute fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
