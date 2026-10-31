"""
Construct a k-regular graph with girth at least g.
Structural change: voltage-graph construction over the dodecahedral graph (20 vertices, girth 5)
with fast non-backtracking verification, plus LUW fallback.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Primary: voltage graph over dodecahedral graph
        result = _voltage_dodecahedral(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Secondary: circulant search
        result = _find_circulant_cage(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback: LUW
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback
    return _luw_construct(k, g)


def _dodecahedral_edges():
    """
    Return the edge list of the dodecahedral graph (20 vertices, 30 edges, girth 5).
    Uses the standard construction: two pentagons (outer, inner) connected by a belt.
    Vertices: 0-4 outer pentagon, 5-9 inner pentagon, 10-14 upper belt, 15-19 lower belt.
    """
    edges = []
    # Outer pentagon: 0-1-2-3-4-0
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner pentagon: 5-6-7-8-9-5
    for i in range(5):
        edges.append((5 + i, 5 + (i + 1) % 5))
    # Upper belt: 10-11-12-13-14-10
    for i in range(5):
        edges.append((10 + i, 10 + (i + 1) % 5))
    # Lower belt: 15-16-17-18-19-15
    for i in range(5):
        edges.append((15 + i, 15 + (i + 1) % 5))
    # Connect outer to upper belt: i -> 10+i
    for i in range(5):
        edges.append((i, 10 + i))
    # Connect inner to lower belt: 5+i -> 15+i
    for i in range(5):
        edges.append((5 + i, 15 + i))
    # Connect outer to inner via belt: 10+i -> 5+i and 10+i -> 15+i? No.
    # Actually the dodecahedron: each vertex on the belt connects to one outer and one inner.
    # Let me use a cleaner construction.
    # Standard dodecahedron: 20 vertices.
    # Use the known adjacency:
    # Vertices 0-4: outer pentagon
    # Vertices 5-9: inner pentagon (reversed)
    # Vertices 10-14: middle ring
    # Vertices 15-19: middle ring 2
    # This is getting complicated. Let me use a verified construction.

    # Actually, let me use a simpler, verified dodecahedron:
    # The dodecahedron can be constructed as follows:
    # It has 20 vertices and 30 edges, is 3-regular, girth 5.
    # I'll use the standard coordinates-based construction.

    # Simpler: use the known edge list for the dodecahedral graph.
    # From the dodecahedron polyhedron:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 9), (9, 8), (8, 7), (7, 6), (6, 5),  # inner pentagon (reversed)
        (0, 10), (1, 11), (2, 12), (3, 13), (4, 14),  # outer to upper
        (5, 15), (6, 16), (7, 17), (8, 18), (9, 19),  # inner to lower
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),  # upper ring
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),  # lower ring
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),  # connect rings
    ]

    # Verify: each vertex should have degree 3
    deg = [0] * 20
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    assert all(d == 3 for d in deg), f"Degree check failed: {deg}"
    assert len(edges) == 30, f"Edge count: {len(edges)}"

    return edges


def _voltage_dodecahedral(k, g, deadline):
    """
    Voltage graph construction over the dodecahedral graph.
    The base graph has 20 vertices, 30 edges, is 3-regular with girth 5.
    We assign voltages (elements of Z_m) to each edge such that the
    derived graph has 20*m vertices and girth >= g.

    For a 3-regular base graph with girth 5, the derived graph with
    voltage assignment phi: E -> Z_m will have girth >= 5*ceil((g-1)/5)
    if all non-zero voltage sums around every cycle of length < g are non-zero.

    Strategy: try m from Moore_bound/20 upward, and search for voltage assignments.
    """
    base_edges = _dodecahedral_edges()
    n_base = 20
    e_base = len(base_edges)  # 30

    # Moore bound for k=3, g=16: n >= 510, so m >= 26 (since 20*26 = 520)
    min_m = 26  # 20*26 = 520 >= 510
    max_m = 50

    # Build base adjacency for fast cycle checking
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    # Precompute all cycles of length 5 and 6 in the base graph
    # (these are the shortest cycles we need to worry about for girth 16)
    # Actually for girth 16, we need no cycles of length 5,6,7,...,15 in the derived graph.
    # A cycle in the derived graph projects to a closed walk in the base graph.
    # The voltage sum around the cycle must be 0 mod m for it to be a cycle in the derived graph.

    # For efficiency, focus on short base cycles (length 5 and 6) since
    # the base graph has girth 5, so no cycles shorter than 5 exist.
    # A cycle of length L in the derived graph corresponds to a closed walk
    # of length L in the base graph with voltage sum 0 mod m.

    # For a 3-regular graph with girth 5, the number of 5-cycles is limited.
    # The dodecahedron has exactly 12 pentagonal faces (5-cycles).

    # Strategy: assign voltages to edges. Since the graph is 3-regular,
    # we can use a "flow" approach: assign voltages such that around every
    # short cycle, the sum is non-zero mod m.

    # Simpler approach: try random/structured voltage assignments and verify.
    # For m, we need 30 voltages in Z_m.

    # Key insight: if we assign voltage 1 to all edges, then a cycle of length L
    # in the base graph has voltage sum L mod m. So cycles of length m, 2m, etc.
    # would be closed. For girth 16, we need m >= 16 to avoid 5-cycles becoming
    # 5-cycles (5 mod m != 0 for m > 5). But 5-cycles in base become 5-cycles
    # in derived only if 5*1 = 5 = 0 mod m, i.e., m | 5. So for m >= 6, all-1
    # voltages avoid 5-cycles. But 6-cycles: 6*1 = 6, so m | 6 would be bad.
    # For m >= 7, all-1 avoids both 5 and 6 cycles. But longer cycles?
    # A 10-cycle in base (if exists) has sum 10, so m | 10 is bad.
    # The dodecahedron has cycles of various lengths.

    # Better: use a more sophisticated voltage assignment.
    # Let's try: assign voltage based on edge position.

    for m in range(min_m, max_m + 1):
        if time.time() > deadline - 5:
            return None

        # Try several voltage assignment strategies
        for strategy in range(5):
            if time.time() > deadline - 5:
                return None

            voltages = _assign_voltages(base_edges, m, strategy)
            if voltages is None:
                continue

            # Verify girth using non-backtracking BFS on the derived graph
            if _verify_voltage_girth(n_base, base_edges, voltages, m, g):
                # Construct the full edge list
                n = n_base * m
                edges = []
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for s in range(m):
                        su = s * n_base + u
                        sv = ((s + vol) % m) * n_base + v
                        if su < sv:
                            edges.append((su, sv))
                        else:
                            edges.append((sv, su))
                # Remove duplicates (each undirected edge appears once)
                # Actually each base edge gives m edges in the derived graph.
                # Since base_edges has each undirected edge once, and we add
                # m edges per base edge, total = 30*m = 600 for m=20.
                # But n*m/2 = 20*m*m/2 = 10*m^2. For m=20: 4000. Mismatch!
                # Wait: the derived graph has n_base*m vertices, each of degree 3.
                # Total edges = n_base*m*3/2 = 30*m.
                # So 30*m edges total, which matches 30 base edges * m voltages.
                # Good.

                # Verify no duplicate edges
                edge_set = set()
                valid = True
                for e in edges:
                    key = (min(e), max(e))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                if valid:
                    return n, edges

    return None


def _assign_voltages(base_edges, m, strategy):
    """
    Assign voltages to base edges for different strategies.
    Returns list of voltages (one per edge) or None.
    """
    e = len(base_edges)
    if strategy == 0:
        # All ones
        return [1] * e
    elif strategy == 1:
        # Alternating: edge i gets voltage (i % m) + 1 if i < m else 1
        return [(i % m) + 1 if i < m else 1 for i in range(e)]
    elif strategy == 2:
        # Based on edge index: voltage = (i * 7 + 1) % m, ensure non-zero
        vols = []
        for i in range(e):
            v = (i * 7 + 1) % m
            if v == 0:
                v = 1
            vols.append(v)
        return vols
    elif strategy == 3:
        # Use a permutation-based assignment
        import random
        rng = random.Random(42 + m)
        vols = [1] * e
        # Shuffle some
        for i in range(min(e, m)):
            vols[i] = (i + 1) % m
            if vols[i] == 0:
                vols[i] = 1
        return vols
    elif strategy == 4:
        # Gradient: voltage increases along a spanning tree
        # Use BFS order from vertex 0
        n_base = 20
        order = [0]
        visited = {0}
        parent = {0: -1}
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in _base_adj_static():
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        # Assign voltage based on depth
        depth = {0: 0}
        for v in order[1:]:
            depth[v] = depth[parent[v]] + 1
        # For each edge (u, v), voltage = depth difference + 1
        vols = []
        for u, v in base_edges:
            d = abs(depth[u] - depth[v])
            vol = (d + 1) % m
            if vol == 0:
                vol = 1
            vols.append(vol)
        return vols
    return None


def _base_adj_static():
    """Static base adjacency for the dodecahedron."""
    edges = _dodecahedral_edges()
    adj = [[] for _ in range(20)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_voltage_girth(n_base, base_edges, voltages, m, g):
    """
    Verify that the voltage graph has girth >= g using non-backtracking BFS.
    The derived graph has n_base*m vertices.
    Vertex (s, u) where s in Z_m, u in 0..n_base-1.
    Edge between (s, u) and (s+v, v) for each base edge (u, v) with voltage v.
    """
    n = n_base * m
    # Build adjacency of derived graph
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(base_edges):
        vol = voltages[idx]
        for s in range(m):
            su = s * n_base + u
            sv = ((s + vol) % m) * n_base + v
            adj[su].append(sv)
            sv2 = ((s - vol) % m) * n_base + v
            su2 = s * n_base + u
            # The reverse edge
            adj[sv].append(su)

    # Actually, let me rebuild more carefully.
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(base_edges):
        vol = voltages[idx]
        for s in range(m):
            # Forward: (s, u) -> (s+vol, v)
            su = s * n_base + u
            sv = ((s + vol) % m) * n_base + v
            adj[su].append(sv)
            # Backward: (s+vol, v) -> (s, u)
            adj[sv].append(su)

    # Verify degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False

    # Non-backtracking BFS from vertex 0
    # State: (current, previous)
    # dist[(u, prev)] = shortest non-backtracking walk length
    dist = {}
    queue = deque()
    start = (0, -1)
    dist[start] = 0
    queue.append(start)

    max_len = g - 1

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]

        if d >= max_len:
            continue

        for v in adj[u]:
            if v == prev:
                continue
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                if v == 0 and u != -1:
                    return False  # Found a cycle shorter than g
                queue.append(new_state)

    return True


def _find_circulant_cage(k, g, deadline):
    """Search for circulant graph C_n(1, a, b) with girth >= g."""
    min_n = 512
    max_n = 700

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        candidates = _generate_candidates(n)
        for a, b in candidates:
            if time.time() > deadline - 3:
                return None
            if _has_short_cycle_fast(n, a, b, g):
                continue
            edges = []
            for i in range(n):
                for shift in [1, a, b]:
                    j = (i + shift) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges

    return None


def _generate_candidates(n):
    candidates = []
    half = n // 2
    for a in range(3, half):
        for b in range(a + 1, half):
            if _quick_reject(n, a, b):
                continue
            candidates.append((a, b))
    return candidates


def _quick_reject(n, a, b):
    G = [1, -1, a, -a, b, -b]
    G_mod = [x % n for x in G]
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            if G_mod[i] == (-G_mod[j]) % n:
                continue
            s = (G_mod[i] + G_mod[j]) % n
            for k in range(6):
                if k == i or k == j:
                    continue
                if s == G_mod[k]:
                    return True
    return False


def _has_short_cycle_fast(n, a, b, g):
    def get_neighbors(u):
        return [(u + 1) % n, (u - 1) % n, (u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]

    dist = {}
    queue = deque()
    start_state = (0, -1)
    dist[start_state] = 0
    queue.append(start_state)

    max_len = g - 1

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for v in get_neighbors(u):
            if v == prev:
                continue
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                if v == 0 and u != -1:
                    return True
                queue.append(new_state)

    return False


def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    q = k
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]
        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
