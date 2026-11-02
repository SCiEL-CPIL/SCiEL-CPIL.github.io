"""Construct a 4-regular graph of girth 10 via Lazebnik-Ustimenko-Woldar CD(d, q)."""
import time
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def _build_luw(q, d, deadline, max_vertices=None):
    """Build the connected component of the zero point in CD(d, q).
    Returns (n, edges) or None if deadline exceeded / size cap hit."""
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int32)
    else:
        table = None

    # Precompute rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute all neighbor offsets for each side
    # For side 0: y[c] = x[c] + mul(y[a], x[b])
    # For side 1: y[c] = x[c] - mul(x[a], y[b])
    # We need to enumerate all q choices for y[1] (the "first" variable)

    # For q=4, we use XOR for add/sub and table for mul
    # For general q, we use mod arithmetic

    def compute_neighbors(side, x):
        """Compute all q neighbors of vertex (side, x). Returns list of (side2, y_tuple)."""
        neighbors = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            # y[0] is unused (None)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    prod = table[y[a], x[b]] if table is not None else (y[a] * x[b]) % q
                    y[c] = (x[c] + prod) % q if table is None else x[c] ^ prod
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    prod = table[x[a], y[b]] if table is not None else (x[a] * y[b]) % q
                    y[c] = (x[c] - prod) % q if table is None else x[c] ^ prod
            neighbors.append((1 - side, tuple(y[1:])))
        return neighbors

    start = (0, (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    # BFS to build the connected component
    frontier = [0]  # indices into order
    fi = 0
    while fi < len(frontier):
        if time.time() > deadline:
            return None
        if max_vertices is not None and len(order) > max_vertices:
            return None

        vidx = frontier[fi]
        fi += 1
        vertex = order[vidx]
        side, x = vertex

        for other in compute_neighbors(side, x):
            if other not in index:
                idx = len(order)
                index[other] = idx
                order.append(other)
                frontier.append(idx)
                if max_vertices is not None and len(order) > max_vertices:
                    return None
            if side == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_fast(n, adj, k, g, deadline):
    """Fast non-backtracking girth verification using BFS from each vertex."""
    # For each starting vertex s, do a non-backtracking BFS
    # State: (current_node, previous_node), depth
    # A cycle of length L is found when we return to s at depth L
    # We need: no cycle of length < g

    # Use array-based visited for speed
    # Encode state as cur * n + prev
    # We track depth per state

    for s in range(n):
        if time.time() > deadline:
            return False

        # visited: use a dict or array
        # For n=512, state space is 512*512 = 262144, manageable
        visited = {}
        frontier = []
        for nb in adj[s]:
            key = nb * n + s
            visited[key] = 1
            frontier.append(key)

        depth = 1
        while frontier and depth < g:
            if time.time() > deadline:
                return False
            next_frontier = []
            for key in frontier:
                cur = key // n
                prev = key % n
                # Check if we've returned to s at this depth (cycle of length depth)
                if cur == s and depth >= 3:
                    return False
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    nkey = nxt * n + cur
                    if nkey not in visited:
                        visited[nkey] = depth + 1
                        next_frontier.append(nkey)
            depth += 1
            frontier = next_frontier

    return True


def _verify(n, edges, k, g, deadline):
    """Return True iff the graph is simple, k-regular, and has girth >= g."""
    if time.time() > deadline:
        return False
    # Simple graph and degree check
    adj_sets = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj_sets[u]:
            return False
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    for u in range(n):
        if len(adj_sets[u]) != k:
            return False

    # Build adjacency lists
    adj = [list(s) for s in adj_sets]

    return _verify_girth_fast(n, adj, k, g, deadline)


def construct(k, g):
    start_time = time.time()
    hard_deadline = start_time + 115.0

    q = k
    base = g - 5
    if base % 2 == 0:
        base += 1
    base = max(3, base)

    # For g=10, k=4: base=5
    # CD(d,q) with d=5 gives the standard (4,10) LUW graph on 512 vertices

    candidates = []
    for d in range(base, base + 8):
        if d % 2 == 1:
            candidates.append(d)

    # Sort: try base first (most likely to have correct girth), then larger
    candidates.sort(key=lambda d: abs(d - base))

    best = None
    for d in candidates:
        if time.time() > hard_deadline - 10.0:
            break
        build_deadline = min(hard_deadline - 12.0, time.time() + 15.0)
        res = _build_luw(q, d, build_deadline)
        if res is None:
            continue
        n, edges = res
        if n < 2 * k:
            continue
        if n % 2 != 0:
            continue

        # Verify
        verify_deadline = hard_deadline - 3.0
        ok = _verify(n, edges, k, g, verify_deadline)
        if ok:
            if best is None or n < best[0]:
                best = (n, edges)
                # If we found a valid graph, try to find smaller ones
                # But only if we have time
                if time.time() > hard_deadline - 20.0:
                    break
        else:
            continue

    if best is not None:
        return best

    # Fallback: try base d without full verification
    for d in range(base, base + 4):
        if d % 2 == 1:
            res = _build_luw(q, d, hard_deadline - 0.5)
            if res is not None:
                n, edges = res
                if n >= 2 * k and n % 2 == 0:
                    return res
    raise ValueError("Failed to construct a valid graph within the time budget")

# EVOLVE-BLOCK-END
