"""Search for a small 5-regular graph with girth >= g.

We generate random k-regular graphs and verify their girth exactly via
non-backtracking walks from every vertex, keeping the smallest verified one.
A deadline guard returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, adj_set, n, k, rng, stubs=None):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    if stubs is None:
        stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj_set[u1].discard(v1)
    adj_set[v1].discard(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    adj_set[u2].discard(v2)
    adj_set[v2].discard(u2)
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj_set[u1].add(v1)
        adj_set[v1].add(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        adj_set[u2].add(v2)
        adj_set[v2].add(u2)
        return False
    if u2 in adj_set[u1] or v2 in adj_set[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj_set[u1].add(v1)
        adj_set[v1].add(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        adj_set[u2].add(v2)
        adj_set[v2].add(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj_set[u1].add(u2)
    adj_set[u2].add(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    adj_set[v1].add(v2)
    adj_set[v2].add(v1)
    return True

# ---------------- exact girth check via non-backtracking walks ----------------
def _girth_ok(adj, g, edge_list=None):
    """Return True if the graph has no cycle shorter than g.
    Uses non-backtracking walks from every vertex up to depth g-1.
    A cycle of length L exists iff there is a non-backtracking walk of length L
    from some vertex back to itself. We check all vertices up to depth g-1.
    """
    n = len(adj)
    if n == 0:
        return True
    if g <= 2:
        return True

    # Build edge list if not provided
    if edge_list is None:
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))

    # Precompute adjacency as numpy arrays for speed
    adj_np = [np.array(adj[u], dtype=np.int32) for u in range(n)]

    max_depth = g - 1  # We need to detect cycles of length < g, i.e., up to g-1

    # Non-backtracking walk state: for each source s, we track walks of length
    # d ending at vertex v having come from vertex p (the previous vertex).
    # State: dict mapping (v, p) -> True if a non-backtracking walk of current
    # length exists from s to v via p.
    #
    # Optimization: use a set of (v, p) pairs. At each step, for each (v, p)
    # in the current frontier, extend to all neighbors w of v where w != p.
    # If w == s and current_length + 1 >= g... wait, we want to detect cycles
    # of length < g. A cycle of length L means a non-backtracking walk of
    # length L from s back to s. So we check at each depth d (1 to g-1)
    # whether s is in the frontier.

    for s in range(n):
        # Frontier: set of (current_vertex, previous_vertex)
        # At depth 1: (v, s) for each neighbor v of s
        frontier = set()
        for v in adj_np[s]:
            frontier.add((int(v), s))

        for depth in range(1, max_depth):
            # Check if s is in the frontier (cycle of length depth)
            # A cycle of length depth < g means girth < g
            for (v, p) in frontier:
                if v == s:
                    return False

            # Extend frontier
            next_frontier = set()
            for (v, p) in frontier:
                for w in adj_np[v]:
                    wi = int(w)
                    if wi != p:
                        next_frontier.add((wi, v))
            frontier = next_frontier
            if not frontier:
                break

    return True

def _girth_ok_fast(adj, g, edge_list=None):
    """Faster girth check using numpy-vectorized non-backtracking walks.
    Returns True if no cycle shorter than g exists."""
    n = len(adj)
    if n == 0:
        return True
    if g <= 2:
        return True

    if edge_list is None:
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))

    # Build adjacency matrix for fast neighbor lookup
    # Use a list of sets for O(1) membership
    adj_set = [set(adj[u]) for u in range(n)]

    max_depth = g - 1

    for s in range(n):
        # Frontier: list of (v, p) pairs
        # Start: neighbors of s
        frontier = [(int(v), s) for v in adj[s]]

        for depth in range(1, max_depth):
            # Check for cycle: is s in any (v, p) pair?
            for (v, p) in frontier:
                if v == s:
                    return False

            # Extend
            next_frontier = []
            for (v, p) in frontier:
                for w in adj[v]:
                    if w != p:
                        next_frontier.append((w, v))
            frontier = next_frontier
            if not frontier:
                break

    return True

def _girth_ok_bfs(adj, g):
    """Girth check via BFS from each vertex. For each source s, BFS up to
    depth floor((g-1)/2). If any vertex is reached at depth d via two
    different parents, or if s is reached at depth < g, we have a short cycle.
    
    Actually the cleanest: for each source s, do BFS. If we find an edge (u,v)
    where dist[u] + dist[v] + 1 < g and the paths don't share the edge, we
    have a cycle. But the non-backtracking approach is more direct.
    
    Let me use the standard approach: BFS from s, track dist and parent.
    For each edge (u,v) not in the BFS tree, if dist[u] + dist[v] + 1 < g,
    we have a cycle of that length.
    """
    n = len(adj)
    if n == 0:
        return True
    if g <= 2:
        return True

    # Precompute edge list
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))

    max_bfs_depth = (g - 1) // 2

    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= max_bfs_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)

        # Check all edges for short cycles
        for u in range(n):
            if dist[u] == -1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    continue
                if u == v:
                    continue
                # Edge (u,v): cycle length is dist[u] + dist[v] + 1
                # but only if the shortest paths from s to u and s to v
                # don't share the edge (u,v) itself.
                # If parent[u] == v or parent[v] == u, the edge is in the
                # BFS tree and the "cycle" is degenerate.
                if parent[u] == v or parent[v] == u:
                    continue
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False

    return True

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_ok_bfs(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # (1) try sizes where girth >= g is plausible
    if k == 5 and g == 9:
        start_n = 100
    else:
        start_n = 2 * k

    n_candidates = []
    n = start_n
    while n <= 600 and time.time() < deadline:
        if (n * k) % 2 != 0:
            n += 1
            continue
        n_candidates.append(n)
        n += 2

    for n in n_candidates:
        if time.time() > deadline:
            break
        adj = None
        for _ in range(200):
            if time.time() > deadline:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        if adj is not None:
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            # Try 2-switch local search
            adj_set = [set(adj[i]) for i in range(n)]
            stubs = array('I', (i for i in range(n) for _ in range(k)))
            check_interval = 50
            for it in range(10000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, adj_set, n, k, rng, stubs):
                    if (it + 1) % check_interval == 0:
                        if consider(n, adj):
                            return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
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
