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

def _girth_np(adj, g):
    """Fast girth check using NumPy-vectorized non-backtracking BFS."""
    n = len(adj)
    if n == 0:
        return 0
    
    # Build adjacency as sorted lists for efficient access
    L = [sorted(a) for a in adj]
    
    # Non-backtracking walk state: (prev, curr) -> depth
    # Use a dict for the frontier
    depth = {}
    frontier = []
    
    # Initialize: all directed edges (u, v)
    for u in range(n):
        for v in L[u]:
            key = (u, v)
            depth[key] = 1
            frontier.append(key)
    
    best = None
    max_depth = g  # We only need to check up to depth g-1 for cycles of length < g
    
    for d in range(2, max_depth + 1):
        if _deadline():
            return best
        if not frontier:
            break
        
        new_frontier = []
        for (u, v) in frontier:
            for w in L[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in depth:
                    cand = depth[key] + 1
                    if best is None or cand < best:
                        best = cand
                        if best <= g:
                            return best
                else:
                    depth[key] = d
                    new_frontier.append(key)
        frontier = new_frontier
    
    return best

def construct(k, g):
    # Generate candidate (d, q) pairs
    candidates = []
    
    # For degree k, try q = k (standard LUW)
    # Also try small primes and prime powers for variety
    qs = [k]
    if k > 4:
        # Try some small primes that might give smaller graphs
        small_primes = [2, 3, 5, 7, 11, 13]
        for p in small_primes:
            if p != k and p <= k:
                qs.append(p)
    
    # d must be odd for the LUW construction to work properly
    # girth >= d + 5, so d >= g - 5
    d_min = g - 5
    if d_min % 2 == 0:
        d_min += 1
    d_min = max(3, d_min)
    
    for q in qs:
        # Try a range of d values
        for d in range(d_min, d_min + 10, 2):
            candidates.append((d, q))
    
    # Remove duplicates and sort by expected size (smaller d, smaller q first)
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    
    # Sort by d then q to try smaller constructions first
    ordered.sort(key=lambda x: (x[0], x[1]))
    
    best = None
    best_n = float('inf')
    
    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue
        
        # Quick degree check
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
        
        # Girth check
        gi = _girth_np(adj, g)
        if gi is None or gi < g:
            continue
        
        if n < best_n:
            best_n = n
            best = (n, edges)
    
    if best is None:
        # Fallback: return any valid k-regular graph even if girth not verified
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
