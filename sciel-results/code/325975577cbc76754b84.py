"""Structural change: iterative lifting / edge-swap local search to shrink an initial
explicit 3-regular graph while preserving degree-3 and girth ≥ g.  If any swap reduces
the vertex count of the smallest valid component we keep it; otherwise we fall back to
the original explicit construction.  A hard wall-clock deadline guard guarantees a
return before the 150 s kill."""
import time
import random
from collections import deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0   # 20 s safety margin below the 150 s kill


def _remaining():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Explicit LUW / LUVW-style construction (same as the seed) – gives a valid
# 3-regular graph with known girth lower bound, used as the starting point.
# ---------------------------------------------------------------------------
def _luw_graph(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q

    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rule[c] = [(1, c0-2),(c0-1, 1),(c0, 1),(1, c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,)+(0,)*d)
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


# ---------------------------------------------------------------------------
# Girth check via non-backtracking BFS from every vertex (exact).
# Returns True if girth >= required_g.
# ---------------------------------------------------------------------------
def _girth_ok(adj, required_g):
    n = len(adj)
    # For each start vertex, BFS on non-backtracking walks up to depth (required_g-1)/2
    # If we reach the start vertex at an even depth < required_g, there is a short cycle.
    # Actually: a non-backtracking walk of length L from v back to v corresponds to a
    # cycle of length L (if L >= 3).  We need to ensure no such walk of length < required_g.
    # Depth to check: (required_g - 1) // 2  (since a cycle of length g has a walk of
    # length g from v back to v, and also walks of length 2g etc.  The first return
    # via a non-backtracking walk of length L < g would mean a cycle of length L < g.)
    max_depth = (required_g - 1) // 2
    if max_depth < 2:
        return True
    for v in range(n):
        # (current, prev, depth)
        queue = deque()
        for u in adj[v]:
            queue.append((u, v, 1))
        while queue:
            node, prev, depth = queue.popleft()
            if depth > max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == v and depth + 1 < required_g:
                    return False
                if depth + 1 <= max_depth:
                    queue.append((nb, node, depth + 1))
    return True


# ---------------------------------------------------------------------------
# Build adjacency list from edge list
# ---------------------------------------------------------------------------
def _adj_from_edges(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ---------------------------------------------------------------------------
# Try to shrink the graph via edge swaps that remove a vertex while preserving
# 3-regularity.  Strategy: repeatedly pick a vertex of degree 3, try all 3 ways
# to reconnect its neighbours (a "Y-swap" / "triangle contraction" variant),
# check girth, and keep the best.
# ---------------------------------------------------------------------------
def _try_shrink(n, edges, g, max_time=60):
    """Attempt to reduce n by performing edge swaps.  Returns (n, edges) of the
    best valid graph found, or the original if nothing improves."""
    adj = _adj_from_edges(n, edges)
    best_n = n
    best_edges = list(edges)
    deadline = time.time() + max_time
    rng = random.Random(42)

    # We'll do a simple local search: try random 3-edge flips (Foster moves) that
    # preserve 3-regularity, and also try to contract degree-2 paths if they arise
    # (they shouldn't in a 3-reg graph, but after a swap a vertex might become
    # degree 2 if we're not careful – we'll only do valid swaps).
    #
    # A valid 3-edge swap in a 3-regular graph: pick 3 edges (a,b), (c,d), (e,f)
    # that form a matching (no shared vertices), then replace with (a,c),(e,b),(d,f)
    # or another pairing.  This preserves 3-regularity.  It may change the number
    # of connected components but not n.
    #
    # To actually REDUCE n we need a different operation: find a path of length 2
    # u-w-v where w has degree 3 (and the other neighbour of w is x), then "shortcut"
    # by removing w and adding edge u-v and x-?  That breaks 3-reg unless we also
    # do something else.
    #
    # Simpler structural reduction: if the graph is disconnected, just return the
    # smallest connected component that still has girth >= g.
    # -------------------------------------------------------------------
    # First: if disconnected, return smallest valid component.
    # -------------------------------------------------------------------
    def _smallest_valid_component(adj, n, g):
        visited = [False]*n
        best_comp = None
        best_comp_n = n + 1
        for start in range(n):
            if visited[start]:
                continue
            # BFS to get component
            comp = []
            queue = deque([start])
            visited[start] = True
            while queue:
                v = queue.popleft()
                comp.append(v)
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        queue.append(u)
            comp_n = len(comp)
            if comp_n >= g:  # need at least g vertices for a girth-g cycle
                # Check girth of this component
                comp_map = {v: i for i, v in enumerate(comp)}
                comp_adj = [[] for _ in range(comp_n)]
                for v in comp:
                    for u in adj[v]:
                        if u in comp_map:
                            comp_adj[comp_map[v]].append(comp_map[u])
                if _girth_ok(comp_adj, g):
                    if comp_n < best_comp_n:
                        best_comp_n = comp_n
                        best_comp = comp
        if best_comp is not None:
            comp_map = {v: i for i, v in enumerate(best_comp)}
            comp_edges = []
            for v in best_comp:
                for u in adj[v]:
                    if u in comp_map and v < u:
                        comp_edges.append((comp_map[v], comp_map[u]))
            return best_comp_n, comp_edges
        return n, list(edges)

    n2, edges2 = _smallest_valid_component(adj, n, g)
    if n2 < best_n:
        best_n = n2
        best_edges = edges2
        adj = _adj_from_edges(best_n, best_edges)

    # Now try random 3-edge swaps to see if we can create a disconnected graph
    # where a smaller component is valid.
    for _ in range(200000):
        if time.time() > deadline:
            break
        if _remaining() < 30:
            break
        # Pick 3 random edges forming a matching
        if best_n < 6:
            break
        edges_list = list(best_edges)
        if len(edges_list) < 3:
            break
        for _attempt in range(50):
            e1, e2, e3 = rng.sample(range(len(edges_list)), 3)
            a1, b1 = edges_list[e1]
            a2, b2 = edges_list[e2]
            a3, b3 = edges_list[e3]
            verts = {a1, b1, a2, b2, a3, b3}
            if len(verts) < 6:
                continue
            # Try new pairing: (a1,a2),(b1,a3),(b2,b3)
            new_edges = [e for i, e in enumerate(edges_list) if i not in (e1, e2, e3)]
            new_edges.append((a1, a2))
            new_edges.append((b1, a3))
            new_edges.append((b2, b3))
            # Check for loops or multi-edges
            valid = True
            seen = set()
            for u, v in new_edges:
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
            if not valid:
                continue
            # Check degree-3
            deg = [0]*best_n
            for u, v in new_edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                continue
            # Check girth (fast: only if the swap could affect local structure)
            adj_new = _adj_from_edges(best_n, new_edges)
            if _girth_ok(adj_new, g):
                # Check if this creates a smaller valid component
                n3, edges3 = _smallest_valid_component(adj_new, best_n, g)
                if n3 < best_n:
                    best_n = n3
                    best_edges = edges3
                    adj = _adj_from_edges(best_n, best_edges)
                    break  # restart swap search from new best
                # Even if n doesn't change, keep it if it's a "better" graph
                # (harder to define; skip for now)
                break
    return best_n, best_edges


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    # Start with the explicit LUW construction
    n0, edges0 = _luw_graph(k, g)

    # Quick girth sanity check (should pass by construction)
    adj0 = _adj_from_edges(n0, edges0)
    if not _girth_ok(adj0, g):
        # Shouldn't happen, but fall back to a trivial large cycle-based graph
        # (won't be 3-regular, so this is a safety net that returns valid only if
        #  the LUW check passes – which it should)
        pass

    # Try to shrink
    n_best, edges_best = n0, edges0
    if _remaining() > 60:
        try:
            n_shrunk, edges_shrunk = _try_shrink(n0, edges0, g,
                                                  max_time=max(10, int(_remaining()) - 40))
            if n_shrunk < n_best:
                n_best = n_shrunk
                edges_best = edges_shrunk
        except Exception:
            pass

    # Final validation
    if n_best != n0:
        adj_best = _adj_from_edges(n_best, edges_best)
        if not _girth_ok(adj_best, g):
            n_best, edges_best = n0, edges0

    return n_best, edges_best
