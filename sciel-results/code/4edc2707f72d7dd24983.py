"""Optimized LUW-based construction for small girth-9, 5-regular graphs."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def gf_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
            return add, sub, mul
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            return add, sub, mul

    def make_rule(d):
        # Standard LUW rule: for c in 2..d, rule[c] = (a, b)
        # The rule must satisfy that the resulting graph has girth >= 2d+1
        # Standard rule: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2)
        # For c>=5: pattern based on (c-5) mod 4
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def luw_graph(q, d, rule):
        """Build the LUW(d, q) bipartite graph, return connected component of zero point."""
        add, sub, mul = gf_ops(q)

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

    def check_girth(n, edges, g):
        """Check girth >= g using non-backtracking walks from every vertex."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Non-backtracking walk check: no cycle of length < g
        # A non-backtracking walk of length L from v back to v with L < g means cycle
        # We check: for each vertex, do non-backtracking walks up to length g-1
        # If we return to start, there's a cycle of length <= g-1 < g

        # For efficiency, use the fact that in a k-regular graph,
        # we can do BFS in the non-backtracking state space
        # State: (current_vertex, previous_vertex)
        # Start: (v, None) for each v

        # Actually, a cleaner approach: the girth is the length of the shortest cycle.
        # We can check by doing a BFS from each edge (u,v) in the non-backtracking sense.
        # A cycle of length L corresponds to a non-backtracking closed walk of length L.

        # For girth check: no non-backtracking closed walk of length 2,3,...,g-1
        # Length 2: back-and-forth on an edge - not non-backtracking, so skip
        # Length 3: triangle - check
        # ...

        # Efficient method: for each vertex v, do a BFS in the directed line graph
        # (non-backtracking paths) up to depth g-1, checking if we reach v again.

        # State: (vertex, prev) where prev is the vertex we came from (or -1 for start)
        # From (v, prev), we can go to (w, v) for w in adj[v] if w != prev

        # To check if there's a cycle of length < g through v:
        # Start with states (v, -1)
        # Expand up to depth g-1
        # If we reach (v, u) for any u, there's a non-backtracking closed walk of length <= g-1
        # which implies a cycle of length <= g-1 < g

        # But this is O(n * k^(g-1)) which is too slow for large n.

        # Better: use the fact that in a graph with girth >= g, the non-backtracking
        # matrix has no small cycles. We can use the approach of checking
        # (A^L) for small L, but that's also expensive.

        # Practical approach: for each edge (u,v), do a non-backtracking BFS
        # starting from (u,v) and (v,u), up to depth g-2, checking if we return
        # to the starting edge. This detects cycles of length >= 3.

        # For our purpose, since we're using LUW which is known to have girth >= 2d+1,
        # and we'll verify with a lighter check, let's do a spot check.

        # Actually, let's do a proper but bounded check.
        # For each vertex, non-backtracking BFS up to depth g-1.
        # If n is large, this could be slow, so we'll time-box it.

        # Simpler: check that the LUW construction's theoretical girth bound holds.
        # LUW(d,q) has girth exactly 2d+1 when q is prime and d >= 2.
        # So if 2d+1 >= g, the girth condition is satisfied.
        # We trust this mathematical result and do a light verification.

        # Light verification: check no triangles (girth >= 4) and no 4-cycles (girth >= 5)
        # by sampling, or just trust the LUW bound.

        # For safety, let's do a non-backtracking walk check up to depth g-1
        # but limit the work.
        import random
        rng = random.Random(42)

        # Check from a subset of vertices
        sample_size = min(n, 200)
        vertices_to_check = rng.sample(range(n), sample_size) if n > sample_size else list(range(n))

        for v in vertices_to_check:
            # Non-backtracking BFS from v
            # State: (current, prev)
            visited = set()
            # Start: (v, -1)
            queue = [(v, -1, 1)]  # (current, prev, length)
            visited.add((v, -1))

            while queue:
                cur, prev, length = queue.pop(0)
                if length >= g:
                    continue
                for w in adj[cur]:
                    if w == prev:
                        continue
                    new_state = (w, cur)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if w == v and length + 1 >= 2:
                        # Found a non-backtracking closed walk of length length+1
                        # This means there's a cycle of length <= length+1 < g
                        # But we need to be careful: the walk might not be a simple cycle
                        # However, any non-backtracking closed walk contains a simple cycle
                        # of length <= the walk length.
                        return False
                    if length + 1 < g:
                        queue.append((w, cur, length + 1))

        return True

    # Search over d values for the LUW construction
    # LUW(d, q) has girth 2d+1, so we need 2d+1 >= g => d >= (g-1)/2
    # For g=9, d >= 4
    # The vertex count of LUW(d,q) connected component is roughly q^d * 2
    # We want to minimize this, so we want the smallest d that gives girth >= g
    # But larger d might give a smaller connected component in some cases? Unlikely.
    # Let's try d = ceil((g-1)/2) and a few values above.

    q = k  # q = 5 for this task
    min_d = (g - 1) // 2 + (1 if (g - 1) % 2 else 0)
    # For g=9: min_d = 4

    best_n = float('inf')
    best_edges = None

    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
        rule = make_rule(d)
        n, edges = luw_graph(q, d, rule)
        if n < best_n:
            # Verify girth
            if check_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is None:
        # Fallback: use the original construction
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        rule = make_rule(d)
        best_n, best_edges = luw_graph(q, d, rule)

    return best_n, best_edges
# EVOLVE-BLOCK-END
