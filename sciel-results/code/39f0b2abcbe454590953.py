"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s kill
    q = k  # 3 for this task

    # Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q = k.
    # For q = 3 (prime), the graph is q-regular bipartite.
    # Girth is at least d + 5 for odd d, d + 4 for even d (roughly).
    # We need girth >= g. So we need d such that the girth bound >= g.
    # For q=3, the LUW graph CD(d,3) has girth at least 2*floor((d+3)/2) or similar.
    # Actually, the known result: CD(d,q) has girth at least 2*floor(d/2)+3 for q odd,
    # or 2*floor(d/2)+4 for q even... Let me just try d values and check.
    #
    # The seed used: d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=16: g-5=11 (odd), so d=11. Let's verify girth is >= 16.
    #
    # Actually, for the LUW construction with q=3, the girth of CD(d,3) is known to be:
    #   - For d odd: girth = d + 5 (if d >= 5)
    #   - For d even: girth = d + 4 (if d >= 4)
    # Wait, that's not quite right either. Let me just compute it.
    #
    # The key insight from the cell findings: "Parameter sweep of LUW construction to find
    # smallest d with girth >= 16, likely reducing n from ~177k to ~6k-20k"
    # This suggests trying smaller d values.
    #
    # Let's try d from 5 upward and check the actual girth by BFS.
    # For q=3, the number of vertices in the connected component of zero is 2 * 3^d roughly?
    # No, it's a bipartite graph where each side has 3^d vertices (vectors in GF(3)^d).
    # So total vertices ~ 2 * 3^d. For d=6: 2*729=1458. For d=7: 2*2187=4374.
    # For d=8: 2*6561=13122. That's manageable.
    #
    # We need girth >= 16. Let's find the smallest d that works.

    def build_luw(d, q):
        """Build LUW graph CD(d, q) for prime q. Returns (n, edges)."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q

        # Rule: for each c in 2..d, rule[c] = (a, b) such that
        # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
        # The rule follows a specific pattern.
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Vertex representation: (side, tuple of length d+1)
        # side 0 = point, side 1 = line
        # Index 0 is None (dummy), indices 1..d are field elements.

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        # y[c] = x[c] - x[a] * y[b]
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        # BFS from zero point to get connected component
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

    def check_girth(n, edges, girth_target):
        """Check if graph has girth >= girth_target using BFS from each vertex.
        Returns True if all cycles have length >= girth_target."""
        from collections import deque
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # For each vertex, BFS to find shortest cycle through it
        # A cycle through vertex v has length = 2*dist + 1 (odd) or 2*dist (even)
        # Actually, we need to find the shortest cycle in the whole graph.
        # Standard approach: for each vertex, do BFS and track the shortest
        # cycle that goes through that vertex.
        #
        # More efficient: for each edge (u,v), remove it and find shortest path
        # from u to v. The cycle length is path_length + 1.
        # But that's O(E * (V+E)) which is too slow.
        #
        # Alternative: BFS from each vertex, and when we encounter a visited node
        # that isn't the parent, we found a cycle. The cycle length is
        # dist[current] + dist[visited] + 1.
        #
        # Actually, the standard way to find girth:
        # For each vertex s, do BFS. When we visit a node v from u, if v was already
        # visited and v != parent_of_u, then we have a cycle of length
        # dist[u] + dist[v] + 1.
        # Take the minimum over all such cycles.
        #
        # But we only need to check if girth >= girth_target, so we can stop BFS
        # at depth (girth_target - 1) // 2.

        max_depth = (girth_target - 1) // 2
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
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle: s -> ... -> u -> v -> ... -> s
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    # Try d values from small to large, find smallest d with girth >= g
    best_result = None
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n, edges = build_luw(d, q)
        if n < 2:
            continue
        # Quick check: is the graph 3-regular?
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        # Check girth
        if check_girth(n, edges, g):
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                # Keep trying smaller d? No, we go from small d up, so first valid is smallest.
                # But wait, smaller d means fewer vertices. So the first d that works is best.
                break

    if best_result is not None:
        return best_result

    # Fallback: use the seed's d value
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = build_luw(d, q)
    return n, edges
# EVOLVE-BLOCK-END
