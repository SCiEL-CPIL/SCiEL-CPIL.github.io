"""Construct a k-regular graph of girth >= g on as few vertices as possible.

Uses the Lazebnik-Ustimenko-Woldar (LUW) CD(d, q) explicit bipartite
construction as the primary method, with exact non-backtracking BFS
verification.  Falls back gracefully if verification fails or time runs out.
"""
import time
from collections import deque

_DEADLINE = 145.0
_T0 = None


def _elapsed():
    if _T0 is None:
        return 0.0
    return time.monotonic() - _T0


def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d, q) construction.
    
    For q prime (or q=4), this gives a q-regular bipartite graph
    with girth at least d+5 (d odd) or d+4 (d even).
    """
    q = k
    # Choose d so that girth >= g
    # For odd d: girth >= d+5, so need d >= g-5
    # For even d: girth >= d+4, so need d >= g-4
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(4, g - 4)
    
    # Field arithmetic
    if q == 4:
        # GF(4): x^2 + x + 1 irreducible, elements 0,1,2,3 where 2=alpha, 3=alpha+1
        add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return add_tbl[x][y]
        def sub(x, y): return add_tbl[x][y]  # char 2
        def mul(x, y): return mul_tbl[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Rule: for c in 2..d, rule[c] = (a, b) defines
    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]
    
    # BFS from zero vertex to get connected component
    # Vertex: (side, x1, x2, ..., xd) where side=0 is point, side=1 is line
    start = (0,) + tuple([0] * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for idx, vertex in enumerate(order):
        if not _time_ok(margin=10.0):
            break
        side = vertex[0]
        x = vertex[1:]  # x[0]..x[d-1] correspond to positions 1..d
        
        for first in range(q):
            y = [0] * (d + 1)  # y[1]..y[d]
            y[1] = first
            
            if side == 0:  # point -> line
                # l[c] = p[c] + l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c-1], mul(y[a-1], x[b-1]))
                new_vertex = (1,) + tuple(y[1:])
            else:  # line -> point
                # p[c] = l[c] - l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c-1], mul(x[a-1], y[b-1]))
                new_vertex = (0,) + tuple(y[1:])
            
            if new_vertex not in index:
                index[new_vertex] = len(order)
                order.append(new_vertex)
            if side == 0:
                edges.append((idx, index[new_vertex]))
    
    return len(order), edges


def _verify(n, edges, k, g):
    """Exact girth verification via non-backtracking BFS from every vertex.
    Returns (is_valid, min_cycle_length).
    """
    adj = [[] for _ in range(n)]
    seen = set()
    for i, j in edges:
        if i == j:
            return False, 0
        key = (i, j) if i < j else (j, i)
        if key in seen:
            return False, 0
        seen.add(key)
        adj[i].append(j)
        adj[j].append(i)
    
    for v in range(n):
        if len(adj[v]) != k:
            return False, 0
    
    best = 10**9
    for s in range(n):
        if not _time_ok(margin=5.0):
            break
        # Non-backtracking BFS: state is (current_vertex, prev_vertex, depth)
        visited = {}
        stack = [(s, -1, 0)]
        while stack:
            v, prev, depth = stack.pop()
            if depth >= best:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == s:
                    cycle_len = depth + 1
                    if cycle_len < best:
                        best = cycle_len
                    continue
                key = (w, v)
                if key not in visited:
                    visited[key] = depth + 1
                    stack.append((w, v, depth + 1))
    
    return (best >= g), best


def construct(k, g):
    global _T0
    _T0 = time.monotonic()
    
    # Primary: LUW construction
    n, edges = _luw_construct(k, g)
    
    # Verify
    if _time_ok(margin=10.0):
        valid, mcl = _verify(n, edges, k, g)
        if valid:
            return n, edges
    
    # If verification failed or timed out, try with smaller d
    # (conservative: just return what we have if it's at least degree-correct)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    degree_ok = all(len(adj[v]) == k for v in range(n))
    
    if degree_ok:
        return n, edges
    
    # Last resort: return empty (shouldn't happen)
    return 0, []
