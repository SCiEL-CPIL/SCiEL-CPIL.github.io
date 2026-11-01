"""Structural redesign: cyclic lift of a small base multigraph to build a
girth-16 3-regular simple graph with as few vertices as possible.

Idea: A 3-regular graph on n = m * L vertices can be seen as a "lift" of a
3-regular base multigraph on m vertices. Each base edge e connects base
vertices (u, v) and, in the lift, connects u_i to v_{i + s_e} (mod L) for
each layer i = 0..L-1, where s_e is a random integer shift in [0, L-1].
By choosing the base graph and the shifts carefully, the resulting lift is
simple (no loops, no parallel edges) and can have girth >= g.

We try small m values and random shift assignments, verifying girth exactly
via non-backtracking walks. A deadline guard ensures we return the best
valid result found before the 150 s limit.
"""

import time
import random
from collections import deque

_DEADLINE = 140.0  # safety margin before 150 s kill


def _build_base_graph(m, rng):
    """Return a 3-regular multigraph on m vertices as a list of (u, v) edges.

    We use a random 3-regular multigraph construction: start with a cycle
    (degree 2 for all), then add m/2 random chords (each chord adds degree 1
    to two vertices). To keep it simple and avoid self-loops, we pick
    distinct vertex pairs.

    Actually, for the lift to work cleanly, the base should be a 3-regular
    simple graph or a controlled multigraph. Let's use a random 3-regular
    simple graph on m vertices (requires m >= 4, m even for a simple 3-regular
    graph to exist... actually 3-regular simple graphs exist for any even m >= 4
    except m=6? No, m=6 has K_{3,3} and others. Let's just use m even >= 8).

    For simplicity and reliability, we'll construct a specific small base:
    a "ladder" or "prism" type graph, or just use a random 3-regular simple graph.
    """
    if m < 4:
        return None
    if m % 2 != 0:
        return None  # 3-regular simple graph requires even m

    # Try to build a random 3-regular simple graph on m vertices
    # using the configuration model, then check simplicity
    for _attempt in range(50):
        # Each vertex has 3 stubs
        stubs = [v for v in range(m) for _ in range(3)]
        rng.shuffle(stubs)
        edges = []
        adj = [set() for _ in range(m)]
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        if ok and all(len(a) == 3 for a in adj):
            return edges
    return None


def _lift_graph(base_edges, L, shifts, m):
    """Build the lifted 3-regular graph.

    base_edges: list of (u, v) base edges
    L: lift factor
    shifts: list of same length as base_edges, shift for each edge
    m: number of base vertices

    Returns: list of edges (i, j) on vertices 0..m*L-1, or None if not simple.
    """
    n = m * L
    edges = []
    seen = set()
    for (u, v), s in zip(base_edges, shifts):
        for i in range(L):
            a = u * L + i
            b = v * L + ((i + s) % L)
            if a == b:
                return None  # loop
            key = (min(a, b), max(a, b))
            if key in seen:
                return None  # parallel edge
            seen.add(key)
            edges.append((a, b))
    # Check degree
    if len(edges) != m * L:  # should be 3 * n / 2 = 3 * m * L / 2... wait
        # Each base edge gives L edges in the lift. Total edges = len(base_edges) * L.
        # For 3-regular on n = m*L vertices: edges = 3*n/2 = 3*m*L/2.
        # So len(base_edges) * L = 3*m*L/2 => len(base_edges) = 3*m/2.
        # That's correct for a 3-regular base.
        pass
    return edges


def _check_girth(adj, g, max_check):
    """Check that no cycle shorter than g exists, using non-backtracking walks.

    adj: list of lists of neighbors
    g: required girth
    Returns True if girth >= g.
    """
    n = len(adj)
    # For each vertex, do a BFS of non-backtracking walks up to depth (g-1)/2
    # Actually, to certify no cycle < g, we need to check that no non-backtracking
    # walk of length < g returns to its start.
    # Use BFS from each (start, prev) state.
    half = (g - 1) // 2
    for start in range(n):
        # BFS: state = (vertex, prev_vertex)
        # We want to find if we can return to start in < g steps without backtracking.
        # Standard: BFS from start, tracking (current, parent) to avoid immediate backtrack.
        visited = {}  # (vertex, prev) -> distance
        dq = deque()
        for nb in adj[start]:
            dq.append((nb, start, 1))
            visited[(nb, start)] = 1
        while dq:
            v, prev, dist = dq.popleft()
            if dist >= g:
                break
            if v == start and dist < g:
                return False
            for nb in adj[v]:
                if nb == prev:
                    continue
                state = (nb, v)
                if state not in visited:
                    visited[state] = dist + 1
                    dq.append((nb, v, dist + 1))
    return True


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    rng = random.Random(42)

    # We focus on k=3, g=16 as per the task.
    # Moore bound for (3, 16): roughly 2 * (3-1)^(g/2 - 1) ... let's just try small m.
    # n = m * L. We want small n. Try small m and moderate L.

    best_n = float('inf')
    best_edges = None

    # Try base graph sizes m from small to larger
    # m must be even and >= 8 for a simple 3-regular graph
    for m in range(8, 40, 2):
        if time.time() - start_time > _DEADLINE:
            break
        base_edges = _build_base_graph(m, rng)
        if base_edges is None:
            continue
        num_base_edges = len(base_edges)  # should be 3*m/2

        # Try different lift factors L
        # n = m * L. We want n as small as possible but >= Moore bound.
        # For g=16, k=3, Moore bound ~ 2 * 2^7 = 256? Let's just try L from 4 upward.
        for L in range(4, 60):
            if time.time() - start_time > _DEADLINE:
                break
            n = m * L
            if n >= best_n:
                break  # no point trying larger L for this m

            # Try multiple random shift assignments
            for attempt in range(30):
                if time.time() - start_time > _DEADLINE:
                    break
                # Assign random shifts, avoid 0 to reduce trivial short cycles
                shifts = [rng.randint(1, L - 1) for _ in range(num_base_edges)]
                edges = _lift_graph(base_edges, L, shifts, m)
                if edges is None:
                    continue
                # Build adjacency
                adj = [[] for _ in range(n)]
                for a, b in edges:
                    adj[a].append(b)
                    adj[b].append(a)
                # Check degree
                if any(len(nei) != k for nei in adj):
                    continue
                # Check girth
                if _check_girth(adj, g, n):
                    best_n = n
                    best_edges = edges
                    break  # found valid for this (m, L), try smaller n next

    if best_edges is None:
        # Fallback: return the seed program's construction
        # (Lazebnik-Ustimenko-Woldar)
        return _seed_construct(k, g)

    return best_n, best_edges


def _seed_construct(k, g):
    """Fallback: the original Lazebnik-Ustimenko-Woldar construction."""
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
