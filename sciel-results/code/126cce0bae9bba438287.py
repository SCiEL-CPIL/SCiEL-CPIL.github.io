"""Cage problem n(k,g): build a k-regular graph with girth >= g on few vertices.

Strategy: try multiple construction families and return the one with fewest
vertices that satisfies all constraints (k-regular, girth >= g, simple).
Families tried in order of expected vertex count:
  1. Known small cages (cubic girth 5 = Petersen, etc.)
  2. Generalized Petersen graphs (3-regular)
  3. Random k-regular graphs (Wormald-Holmes) with girth check
  4. LUW algebraic construction as fallback
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    deadline = time.time() + 110.0
    best_n = None
    best_edges = None

    def check_girth(n, edges, g):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in adj:
            u.sort()
        # Non-backtracking BFS: state = (vertex, prev_vertex)
        # We look for cycles of length < g
        # A non-backtracking walk of length L from v that returns to v gives a cycle
        # Actually, we need: no cycle of length < g.
        # Use: for each vertex, BFS in the non-backtracking cover, check if we can
        # return to start within < g steps.
        # More efficient: check that the graph has no cycle of length < g.
        # We do BFS from each vertex tracking (current, prev) to avoid backtracking.
        # If we reach (start, some_prev) at depth < g, there's a short cycle.
        # But this is O(n^2 * k) which may be too slow for large n.
        # Alternative: use the fact that in a k-regular graph, a cycle of length c
        # corresponds to a closed non-backtracking walk of length c.
        # For our purposes with the small graphs we'll generate, this is fine.

        # For efficiency with potentially larger graphs, use a smarter check:
        # BFS from each vertex in the non-backtracking sense up to depth g-1.
        # If we revisit the start vertex, we found a cycle shorter than g.
        max_depth = g - 1
        for start in range(n):
            # visited: (vertex, prev) -> depth
            visited = {}
            queue = deque()
            for nb in adj[start]:
                visited[(nb, start)] = 1
                queue.append((nb, start))
            while queue:
                v, prev = queue.popleft()
                d = visited[(v, prev)]
                if d >= max_depth:
                    continue
                for nb in adj[v]:
                    if nb == prev:
                        continue  # non-backtracking
                    if nb == start:
                        # Found a cycle of length d+1 < g
                        return False
                    if (nb, v) not in visited:
                        visited[(nb, v)] = d + 1
                        queue.append((nb, v))
        return True

    def is_valid(n, edges, k, g):
        if n < 2:
            return False
        # Check degree
        deg = [0] * n
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        if not check_girth(n, edges, g):
            return False
        return True

    def try_candidate(n, edges):
        nonlocal best_n, best_edges
        if best_n is not None and n >= best_n:
            return
        if is_valid(n, edges, k, g):
            best_n = n
            best_edges = edges

    # --- Family 1: Known small cages ---
    # Petersen graph: 3-regular, girth 5, 10 vertices
    if k == 3 and g <= 5:
        # Petersen graph
        n = 10
        edges = []
        # Outer pentagon: 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner star: 5-7-9-6-8-5
        inner = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner[i], inner[(i + 2) % 5]))
        # Spokes
        for i in range(5):
            edges.append((i, 5 + i))
        try_candidate(n, edges)

    # Heawood graph: 3-regular, girth 6, 14 vertices
    if k == 3 and g <= 6:
        n = 14
        edges = []
        # Triangular prism x 2? No, Heawood is the incidence graph of Fano plane.
        # 7 points, 7 lines, bipartite
        # Points 0-6, Lines 7-13
        fano_lines = [
            [0, 1, 2], [0, 3, 4], [0, 5, 6],
            [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
        ]
        for line_idx, line in enumerate(fano_lines):
            for pt in line:
                edges.append((pt, 7 + line_idx))
        try_candidate(n, edges)

    # Pappus graph: 3-regular, girth 4... no, girth 4. Skip.
    # Desargues graph: 3-regular, girth 6, 20 vertices
    if k == 3 and g <= 6:
        # Desargues: bipartite, 10+10
        n = 20
        edges = []
        # Vertices 0-9 (points), 10-19 (lines)
        # Desargues configuration: 10 points, 10 lines, 3-3
        # Point i connects to lines... let me use the standard construction
        # Actually let me skip this and rely on generalized Petersen
        pass

    # --- Family 2: Generalized Petersen graphs (3-regular) ---
    if k == 3:
        # GP(n, k): 2n vertices, girth depends on n and k
        # For girth >= 6: need n >= 7, k not creating short cycles
        for n in range(5, 200):
            if time.time() > deadline:
                break
            if best_n is not None and 2 * n >= best_n:
                break
            edges = []
            # Outer cycle: 0..n-1
            for i in range(n):
                edges.append((i, (i + 1) % n))
            # Inner star: n..2n-1
            for i in range(n):
                edges.append((n + i, n + (i + k) % n))
            # Spokes
            for i in range(n):
                edges.append((i, n + i))
            try_candidate(2 * n, edges)

    # --- Family 3: Random k-regular graphs ---
    # Start from Moore bound + small margin
    def moore_bound(k, g):
        if k == 2:
            return g if g % 2 == 0 else g + 1
        if g % 2 == 0:
            # Even girth g: n >= k + (k-1)(k^((g/2)-1) - 1)/(k-2)
            # Actually: n >= 2 * (k-1) * (k^((g/2)-1) - 1) / (k-2) + 2
            # Standard: n >= k + (k-1) * sum_{i=0}^{(g/2)-2} (k-1)^i * 2
            # Let me use: for even g=2r, n >= 2*(k-1)*(k^(r-1)-1)/(k-2) + 2
            r = g // 2
            if k == 2:
                return 2 * r
            s = 0
            for i in range(r - 1):
                s += (k - 1) ** i
            return 2 * (k - 1) * s // (k - 2) + 2
        else:
            # Odd girth g=2r+1: n >= k + (k-1) * sum_{i=0}^{r-1} (k-1)^i
            r = (g - 1) // 2
            s = 0
            for i in range(r):
                s += (k - 1) ** i
            return k + (k - 1) * s

    mb = moore_bound(k, g)
    # Try random regular graphs starting near Moore bound
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        # Try a range of sizes
        # Start from mb, increase gradually
        n_try = mb + attempt * max(1, mb // 50)
        if n_try > 5000:  # safety cap
            n_try = 5000
        if best_n is not None and n_try >= best_n:
            continue

        # Generate random k-regular graph on n_try vertices
        # Using configuration model with rejection
        if k * n_try % 2 != 0:
            n_try += 1
            if best_n is not None and n_try >= best_n:
                continue

        graph = None
        for inner_attempt in range(20):
            if time.time() > deadline:
                break
            # Configuration model
            half_edges = k * n_try // 2
            stubs = []
            for v in range(n_try):
                stubs.extend([v] * k)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                # Check degree (should be k by construction)
                graph = (n_try, edges)
                break

        if graph is None:
            continue
        try_candidate(graph[0], graph[1])

    # --- Family 4: LUW construction as fallback ---
    def field_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def build_luw(q, d):
        add, sub, mul = field_ops(q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if best_n is None:
        q = k
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1
        n_luw, edges_luw = build_luw(q, d)
        try_candidate(n_luw, edges_luw)

    if best_n is None:
        # Absolute fallback: cycle (only valid for k=2)
        if k == 2:
            n = g if g % 2 == 0 else g + 1
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        # Should not reach here, but return something
        return 0, []

    return best_n, best_edges

# EVOLVE-BLOCK-END
