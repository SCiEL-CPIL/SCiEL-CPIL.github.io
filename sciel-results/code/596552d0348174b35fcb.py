import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 105.0

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
            
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _girth(adj, g):
    n = len(adj)
    if n == 0:
        return 0
    L = [sorted(a) for a in adj]
    depth = {}
    q = deque()
    for u in range(n):
        for v in L[u]:
            q.append((u, v))
            depth[(u, v)] = 1
    best = None
    while q:
        if _deadline():
            return best
        u, v = q.popleft()
        d = depth[(u, v)]
        for w in L[v]:
            if w == u:
                continue
            if (v, w) in depth:
                cand = depth[(v, w)] + 1
                if best is None or cand < best:
                    best = cand
                if best <= g:
                    return best
            else:
                depth[(v, w)] = d + 1
                q.append((v, w))
    return best

def construct(k, g):
    candidates = []
    if k == 4:
        qs = [4]
    else:
        qs = [k]
    for q in qs:
        d0 = g - 5
        if d0 % 2 == 0:
            d0 += 1
        d0 = max(3, d0)
        for d in (d0, d0 + 2, d0 + 4):
            candidates.append((d, q))
            
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    best = None
    best_n = float('inf')
    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue
            
        deg = [0] * n
        seen_edges = set()
        valid = True
        for (i, j) in edges:
            if i == j or i < 0 or i >= n or j < 0 or j >= n:
                valid = False
                break
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_edges:
                valid = False
                break
            seen_edges.add((a, b))
            deg[a] += 1
            deg[b] += 1
        if not valid or not all(x == k for x in deg):
            continue
            
        gi = _girth(adj, g)
        if gi is None or gi < g:
            continue
            
        if n < best_n:
            best_n = n
            best = (n, edges)
            
    if best is None:
        for (d, q) in ordered:
            try:
                n, edges, adj = _luw_graph(d, q)
                deg = [0] * n
                for (i, j) in edges:
                    deg[i] += 1
                    deg[j] += 1
                if all(x == k for x in deg):
                    return n, edges
            except Exception:
                continue
        return 0, []
        
    return best
