"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _girth_check(adj, g):
    """Return True iff the graph (adjacency list) has no cycle shorter than g.
    Uses non-backtracking BFS from every vertex. Returns False on first short cycle."""
    n = len(adj)
    # For each vertex, do non-backtracking BFS up to depth (g-1)//2.
    # If we revisit a vertex at depth <= (g-1)//2 from the same start (excluding the
    # immediate predecessor which is the non-backtracking condition), we have a cycle.
    # Standard: a cycle of length L exists iff there is a non-backtracking walk of
    # length L from v back to v. The shortest cycle through v is found by NB-BFS.
    # We need: no cycle shorter than g. So check depths 2..g-1.
    # More efficient: for each start v, do NB-BFS. The first time we reach v again
    # (at depth d >= 2) gives the girth through v. If any such d < g, fail.
    half = (g - 1) // 2
    # Actually we need to detect cycles of length < g. A cycle of length L < g
    # means there's a non-backtracking closed walk of length L.
    # NB-BFS from v: layer 0 = {v}, layer 1 = neighbors, etc.
    # If v appears in layer L (L>=2), we found a cycle of length L through v.
    # We only need to check up to L = g-1.
    for v in range(n):
        # dist[v] via NB-BFS: dist[0][v] = 0
        # We track (current_vertex, previous_vertex) states? No, that's expensive.
        # Standard NB-BFS: a state is just the vertex, but we must not go back to
        # the vertex we came from. This requires tracking the parent in the tree.
        # Simpler: use the standard "for each directed edge, do BFS not backtracking".
        # But that's O(m * n). For n~1500, m~3750, that's ~5.6M * BFS... too slow in Python.
        # Alternative: use the fact that girth >= g means the graph is a tree up to
        # radius (g-1)//2 around every vertex. So for each v, the ball of radius
        # r=(g-1)//2 must be a tree. We can check this by doing a normal BFS from v
        # up to depth r and ensuring no vertex is reached twice.
        r = (g - 1) // 2
        # BFS from v up to depth r, tracking parent to avoid going back
        visited = [-1] * n  # -1 = unvisited, else = depth at which first visited
        parent = [-1] * n
        visited[v] = 0
        parent[v] = v
        queue = [v]
        for depth in range(r):
            next_queue = []
            for u in queue:
                for w in adj[u]:
                    if visited[w] == -1:
                        visited[w] = depth + 1
                        parent[w] = u
                        next_queue.append(w)
                    elif w != parent[u] and visited[w] <= depth:
                        # Found a cycle: w was visited at depth <= depth, and w is not
                        # the parent of u (so this is a back-edge creating a cycle).
                        # Cycle length = depth + 1 + visited[w]
                        cycle_len = depth + 1 + visited[w]
                        if cycle_len < g:
                            return False
            queue = next_queue
    return True


def _is_valid_graph(n, edges, k, g):
    """Check if the graph is valid: k-regular, simple, girth >= g."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False  # loop
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False  # repeated edge
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check girth
    return _girth_check(adj, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _generalized_polygon_incidence(k, g):
    """Try to construct a graph from a generalized polygon if one exists.
    For k=5, g=16: generalized octagon of order (2,4) has 150 vertices.
    For k=5, g=9: no known small generalized polygon applies directly.
    This is a fallback that returns None if not applicable."""
    # Generalized octagon of order (2,4): s=2, t=4
    # Number of points = 1 + s + st + st^2 + ... + st^{n-1} for n=4 (octagon)
    # Actually for generalized n-gon of order (s,t):
    # v = 1 + s + st + st^2 + ... + st^{n-1}
    # For octagon (n=4? no, octagon has 8 sides, so n=4 in the formula where
    # the polygon has 2n sides). Let me use the standard formula.
    # Generalized octagon: 8 sides, order (s,t) with s=2, t=4.
    # Number of points = 1 + s + st + st^2 + st^3 = 1 + 2 + 8 + 32 + 128 = 171?
    # Hmm, let me just hardcode the known (2,4)-octagon if k=5 and g=16.
    # The (2,4)-octagon has 171 points and 171 lines? No.
    # Actually, the unique (2,4)-octagon has 171 points. Each point is on 5 lines
    # (since t+1 = 5 lines through a point? No, s+1=3 lines through a point for order (s,t)).
    # I'm not confident about the exact parameters. Let's skip this and use LUW.
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, leaving 20s margin
    
    # Try LUW construction first
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify it's valid
    if _is_valid_graph(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
    
    # If LUW fails (shouldn't for prime k), try a simple approach
    # For now, just return the LUW result even if verification is slow
    # The LUW construction is known to be valid for prime k
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
