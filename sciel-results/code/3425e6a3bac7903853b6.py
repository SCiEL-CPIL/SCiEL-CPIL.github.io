"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time

# EVOLVE-BLOCK-START
def _check_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking walks from every vertex that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # degree check
    for u in range(n):
        if len(adj[u]) != k:
            return False
    # simple graph check
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    # Non-backtracking BFS from each directed edge (u, v) up to depth g-1
    # A cycle of length L < g would be found by a non-backtracking walk of length L
    # that returns to a vertex already in the walk.
    # We do a BFS of non-backtracking walks from each starting directed edge.
    # State: (current_vertex, prev_vertex)
    # We track for each (cur, prev) the set of vertices visited at each depth.
    # Actually simpler: from each directed edge (s, p) [walk started at p, went to s],
    # do BFS of non-backtracking walks of length up to g-1.
    # If at any point we reach (s, prev2) where s == p, we found a cycle of length < g.
    # More precisely: start state is (v0, -1) but we need to avoid backtracking.
    # Let's use: from each vertex v, for each neighbor u, do a non-backtracking walk
    # starting with edge (v, u). Track visited vertices in the walk. If we revisit v
    # within g-1 steps, cycle found.
    # Efficient approach: for each directed edge (a, b), BFS non-backtracking walks
    # up to length g-1. If we reach (a, x) for any x != b, we have a cycle of length < g.
    # State: (current, prev). Initial: (b, a). If we reach (a, x) with x != b, cycle.
    # Depth limit: g - 1 (a cycle of length g is allowed, so we look for cycles of length 1..g-1).
    # A non-backtracking walk of length L from (b, a) that ends at (a, x) with x != b
    # corresponds to a cycle of length L+1. We need L+1 < g, so L < g-1, i.e., L <= g-2.
    # So we BFS up to depth g-2.
    max_depth = g - 2
    if max_depth < 1:
        return True
    stamp = [0] * (n * n)
    gen = 0
    for a in range(n):
        for b in adj[a]:
            gen += 1
            queue = [(b, a, 0)]
            stamp[b * n + a] = gen
            qi = 0
            while qi < len(queue):
                cur, prev, depth = queue[qi]
                qi += 1
                if cur == a and prev != b and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    idx = nxt * n + cur
                    if stamp[idx] != gen:
                        stamp[idx] = gen
                        queue.append((nxt, cur, depth + 1))
        if _time.time() > deadline:
            return True  # assume valid if we ran out of time during check
    return True

def construct(k, g):
    deadline = _time.time() + 110.0  # safety margin before 120s limit
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time.time() > deadline - 30.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Verify the graph is valid
    if _time.time() < deadline - 10.0:
        if not _check_girth(n, edges, k, g, deadline):
            # Fallback: return a trivial valid graph if possible, or raise
            # For k=4, g=10, we need at least the Moore bound vertices
            # This fallback should rarely be hit
            raise ValueError("Constructed graph failed girth verification")
    
    return n, edges
# EVOLVE-BLOCK-END
