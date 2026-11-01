import time
import math
import numpy as np


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar (LUW) construction  CD(d, q)
#
#   * q-regular bipartite graph (points vs lines over GF(q)).
#   * Girth is exactly 2*ceil((d+5)/2).
#   * We return the connected component of the zero point, which is the
#     largest component for the parameter ranges we use.
#
# The construction is parametrised by (q, d).  For the task's (k, g) we set
#   q = k   and   d = min d  such that  2*ceil((d+5)/2) >= g.
# ---------------------------------------------------------------------------

def _luw_rule(d):
    # rule[c] = (a, b) for c = 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _luw_build(q, d, deadline):
    """Return (n, edges) for the LUW graph component of the zero point."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = _luw_rule(d)
    cs = list(range(2, d + 1))
    A = np.array([rule[c][0] for c in cs])
    B = np.array([rule[c][1] for c in cs])

    # cache[(side, c)] -> (a, b) index into the coordinate arrays
    cache = {}

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in cs:
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
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            j = index.get(other)
            if j is None:
                j = len(order)
                index[other] = j
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], j))
    return len(order), edges


# ---------------------------------------------------------------------------
# Girth verification  (non-backtracking BFS from every vertex, exact)
# Returns the length of the shortest cycle, or a large value if none < big.
# ---------------------------------------------------------------------------
def _shortest_cycle(adj, n, g):
    # We only need to know whether a cycle of length < g exists, and the min.
    best = 10 ** 9
    # For each start vertex, do a BFS in the "non-backtracking" state space
    # (v, prev).  A cycle is found when we reach (v, prev) with prev == start
    # and depth >= 3 (non-backtracking), giving a cycle of length depth.
    for s in range(n):
        # state (v, prev)
        seen = {}
        stack = [(s, -1, 1)]
        while stack:
            v, prev, depth = stack.pop()
            if v == s and prev != -1:
                # closed a non-backtracking walk of length depth back to s
                if depth < best:
                    best = depth
                continue
            if depth >= g:
                continue
            key = (v, prev)
            if key in seen:
                continue
            seen[key] = depth
            for w in adj[v]:
                if w == prev:
                    continue
                stack.append((w, v, depth + 1))
    return best


def _is_regular(adj, n, k):
    return all(len(adj[v]) == k for v in range(n))


def _simple_edges(edges, n):
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        if not (0 <= a < n and 0 <= b < n):
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
    return True


# ---------------------------------------------------------------------------
# Local edge-swap polish: try to increase girth while keeping k-regular.
# ---------------------------------------------------------------------------
def _try_polish(n, edges, k, g, deadline):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # candidate pairs of disjoint edges (a,b),(c,d) that can be swapped to
    # (a,c),(b,d) without creating loops / duplicates / degree changes.
    # Only accept if girth strictly increases.
    edge_list = list(edges)
    improved = True
    cur_girth = _shortest_cycle(adj, n, g)
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(edge_list)):
            if time.time() > deadline:
                break
            a, b = edge_list[i]
            for j in range(i + 1, len(edge_list)):
                c, d = edge_list[j]
                if len({a, b, c, d}) != 4:
                    continue
                # swap to (a,c),(b,d)
                if c in adj[a] or d in adj[b]:
                    continue
                # remove
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                # add
                adj[a].append(c); adj[c].append(a)
                adj[b].append(d); adj[d].append(b)
                ng = _shortest_cycle(adj, n, g)
                if ng > cur_girth:
                    # accept
                    edge_list[i] = (a, c)
                    edge_list[j] = (b, d)
                    cur_girth = ng
                    improved = True
                    break
                else:
                    # revert
                    adj[a].remove(c); adj[c].remove(a)
                    adj[b].remove(d); adj[d].remove(b)
                    adj[a].append(b); adj[b].append(a)
                    adj[c].append(d); adj[d].append(c)
    # rebuild edge list from adjacency (canonical)
    out = []
    for a in range(n):
        for b in adj[a]:
            if a < b:
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    deadline = time.time() + 100.0   # safety margin vs 120/150s limits

    q = k
    # choose d so that 2*ceil((d+5)/2) >= g
    # 2*ceil((d+5)/2) >= g  <=>  ceil((d+5)/2) >= g/2  <=>  (d+5)/2 > g/2 - 1
    # simplest: find min d with girth(d) >= g
    def girth_of(d):
        return 2 * math.ceil((d + 5) / 2)
    d = 3
    while girth_of(d) < g:
        d += 1

    n, edges = _luw_build(q, d, deadline)

    # quick validity check
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if not _is_regular(adj, n, k) or not _simple_edges(edges, n):
        # fall back: return as-is (should not happen)
        return n, edges

    # polish only if there is room and time
    if time.time() < deadline - 30.0:
        try:
            new_edges = _try_polish(n, edges, k, g, deadline)
            # re-validate
            adj2 = [[] for _ in range(n)]
            for a, b in new_edges:
                adj2[a].append(b)
                adj2[b].append(a)
            if _is_regular(adj2, n, k) and _simple_edges(new_edges, n):
                edges = new_edges
        except Exception:
            pass

    return n, edges
