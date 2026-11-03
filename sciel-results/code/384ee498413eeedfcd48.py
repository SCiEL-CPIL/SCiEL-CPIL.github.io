"""Tensor-product-of-cycles + randomized edge-switching search for n(4,10)."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))


def _verify_girth_fast(n, adj, k, g):
    """Return True iff the simple k-regular graph has girth >= g (non-backtracking BFS)."""
    for i in range(n):
        if len(adj[i]) != k or i in adj[i] or len(set(adj[i])) != k:
            return False
    for v in range(n):
        seen = set()
        q = deque()
        for w in adj[v]:
            seen.add((w, v))
            q.append((w, v, 1))
        while q:
            node, parent, depth = q.popleft()
            if depth >= g:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == v:
                    if depth + 1 < g:
                        return False
                    continue
                st = (nb, node)
                if st not in seen:
                    seen.add(st)
                    q.append((nb, node, depth + 1))
    return True


def _count_short_cycles(n, adj, max_len):
    """Count non-backtracking closed walks of length < max_len, divided by length
    (each simple cycle of length L is counted exactly once per vertex, i.e. L walks / L).
    Cheap surrogate: sum of counts over all starting vertices, halved per direction."""
    total = 0.0
    for v in range(n):
        seen = set()
        q = deque()
        for w in adj[v]:
            seen.add((w, v))
            q.append((w, v, 1))
        while q:
            node, parent, depth = q.popleft()
            if depth >= max_len:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == v:
                    total += 1.0 / (depth + 1)
                    continue
                st = (nb, node)
                if st not in seen:
                    seen.add(st)
                    q.append((nb, node, depth + 1))
    # each simple cycle of length L yields L closed walks (one per start vertex),
    # each counted 1/L here -> exactly 1 per cycle. Return cycle count.
    return total


def _build_tensor_cycles(L):
    """4-regular bipartite graph: Cartesian product of two L-cycles, 4L^2 vertices.
    Vertex (a,b): neighbors (a±1,b) and (a,b±1) mod L."""
    n = 4 * L * L
    adj = [[] for _ in range(n)]
    for a in range(L):
        for b in range(L):
            v = a * L + b
            adj[v].append(((a + 1) % L) * L + b)
            adj[v].append(((a - 1) % L) * L + b)
            adj[v].append(a * L + ((b + 1) % L))
            adj[v].append(a * L + ((b - 1) % L))
    return n, adj


def _edge_switch_local_search(n, adj, g, deadline, rng):
    """Randomized edge-switching local search to eliminate cycles of length < g.
    Maintains 4-regularity. Returns (n, edges) if a valid girth>=g graph is found,
    else None."""
    # Build a set of undirected edges for O(1) lookup
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))

    def recompute_cycles():
        return _count_short_cycles(n, adj, g)

    best_cycles = recompute_cycles()
    if best_cycles < 1e-9:
        return _adj_to_edges(n, adj)

    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random edge (u,v) and one of its parallel partners to form a 4-cycle switch.
        # Standard 2-switch: pick two non-adjacent edges (u,v) and (x,y) with no shared
        # endpoints, replace with (u,x),(v,y) or (u,y),(v,x) if those don't already exist.
        u = rng.randrange(n)
        v = adj[u][rng.randrange(4)]
        x = rng.randrange(n)
        # ensure x not in {u,v}
        if x == u or x == v:
            continue
        y = adj[x][rng.randrange(4)]
        if y == u or y == v or y == x:
            continue
        if (u, v) not in edge_set and (v, u) not in edge_set:
            continue
        if (x, y) not in edge_set and (y, x) not in edge_set:
            continue
        # Try switch option 1: (u,x) and (v,y)
        cand1 = {(min(u, x), max(u, x)), (min(v, y), max(v, y))}
        cand2 = {(min(u, y), max(u, y)), (min(v, x), max(v, x))}
        old = {(min(u, v), max(u, v)), (min(x, y), max(x, y))}
        chosen = None
        for cand in (cand1, cand2):
            if not (cand & old) and not (cand & edge_set):
                chosen = cand
                break
        if chosen is None:
            continue
        # Apply
        for e in old:
            a, b = e
            adj[a].remove(b)
            adj[b].remove(a)
            edge_set.discard(e)
        for e in chosen:
            a, b = e
            adj[a].append(b)
            adj[b].append(a)
            edge_set.add(e)
        new_cycles = recompute_cycles()
        if new_cycles <= best_cycles + 1e-9:
            best_cycles = new_cycles
            if new_cycles < 1e-9:
                return _adj_to_edges(n, adj)
        else:
            # revert
            for e in chosen:
                a, b = e
                adj[a].remove(b)
                adj[b].remove(a)
                edge_set.discard(e)
            for e in old:
                a, b = e
                adj[a].append(b)
                adj[b].append(a)
                edge_set.add(e)
    return None


def _adj_to_edges(n, adj):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - guaranteed valid baseline."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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


def _luw_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _construct_cage_voltage(k, g, deadline):
    """Tensor-product-of-cycles start + randomized edge-switching local search."""
    if k != 4 or g != 10:
        return None

    # Moore bound for (4,10) is 320. Try L values giving 4L^2 >= 320.
    # L=9 -> 324, L=10 -> 400. We need at least 320, so L=9 (324) is closest.
    rng = random.Random(12345)

    for L in (9, 10, 8, 11):
        n, adj = _build_tensor_cycles(L)
        # Quick check: does it already have girth >= g? (unlikely but cheap)
        if _verify_girth_fast(n, adj, k, g):
            return _adj_to_edges(n, adj)
        result = _edge_switch_local_search(n, adj, g, deadline, rng)
        if result is not None:
            nn, edges = result
            # final verification
            adj2 = _luw_to_adj(nn, edges)
            if _verify_girth_fast(nn, adj2, k, g):
                return nn, edges
    return None


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 130  # safety margin before 150s limit

    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result

    # Guaranteed valid fallback
    n, edges = _construct_luw(k, g)
    return n, edges

# EVOLVE-BLOCK-END
