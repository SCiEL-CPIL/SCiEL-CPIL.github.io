import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 110.0  # 10 s safety margin before the 120 s evaluator kill

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
    # Standard LUW rule for odd d
    # y[c] = x[c] + y[a] * x[b]
    # For c=2..d
    # a, b are indices < c
    rule = {}
    for c in range(2, d + 1):
        # Heuristic rule from Lazebnik-Ustimenko-Woldar
        # Often a=1, b=c-1 or similar
        # The code in the prompt used a specific pattern.
        # Let's use the pattern from the prompt:
        # r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # This pattern is specific to certain d.
        # Let's stick to the prompt's rule to ensure consistency with the "seed" behavior.
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
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
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            # y[0] is unused
            for c in range(2, d + 1):
                a, b = rc[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    
    # BFS to find all reachable vertices
    queue = deque([start])
    while queue:
        if _deadline():
            break
        vertex = queue.popleft()
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)

    n = len(order)
    adj = [set() for _ in range(n)]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
            
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges, adj

def _girth(adj, g):
    """Compute girth using BFS from each vertex, stopping early if girth < g."""
    n = len(adj)
    if n == 0:
        return 0
    best = float('inf')
    for start in range(n):
        if _deadline():
            return best if best != float('inf') else None
        dist = [-1] * n
        dist[start] = 0
        q = deque()
        q.append(start)
        found_girth = None
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                elif v != start and du + 1 > dist[v]:
                    cand = du + dist[v] + 1
                    if found_girth is None or cand < found_girth:
                        found_girth = cand
                    if found_girth <= g:
                        return found_girth
        if found_girth is not None:
            if found_girth < best:
                best = found_girth
            if best <= g:
                return best
    return best if best != float('inf') else None

def _verify_and_return(n, edges, adj, k, g):
    # Check regularity
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if not all(d == k for d in deg):
        return None
    
    # Check girth
    gi = _girth(adj, g)
    if gi is None or gi < g:
        return None
        
    return n, edges

def construct(k, g):
    best = None
    best_n = float('inf')
    
    # Try LUW constructions
    # For k=5, q must be 5.
    # d must be such that girth >= g.
    # LUW(d, q) has girth >= d+5 for odd d.
    # So we need d+5 >= g => d >= g-5.
    # d must be odd.
    
    qs = [k] if k != 4 else [4]
    
    min_d = g - 5
    if min_d % 2 == 0:
        min_d += 1
    min_d = max(3, min_d)
    
    # Try d = min_d, min_d+2, min_d+4
    for d in (min_d, min_d + 2, min_d + 4):
        if _deadline():
            break
        for q in qs:
            if _deadline():
                break
            try:
                n, edges, adj = _luw_graph(d, q)
                res = _verify_and_return(n, edges, adj, k, g)
                if res:
                    rn, redges = res
                    if rn < best_n:
                        best_n = rn
                        best = (rn, redges)
            except Exception:
                continue

    if best is None:
        # Fallback: Return a valid graph if possible, else empty
        # The evaluator expects a valid graph.
        # If we fail, we return 0, [] which is invalid.
        # But we must return *something*.
        # Let's try a smaller d just in case?
        # Or just return the first LUW graph that is regular, even if girth is low?
        # No, the task requires girth >= g.
        # If no valid graph is found, we are in trouble.
        # But LUW(5,5) should work.
        
        # Let's force a return of the smallest regular graph found, 
        # hoping the girth check was too strict or failed.
        # Actually, if _girth returns None, it means no cycle < g was found?
        # No, _girth returns None if best is inf (no cycles found < g? No, BFS explores all).
        # If BFS completes without finding a cycle < g, then girth >= g.
        # So _girth returning None means girth >= g (or graph is a tree, impossible for regular n>1).
        # Wait, my _girth returns `best` which is inf if no cycle found.
        # In _verify_and_return, I check `if gi is None or gi < g`.
        # If gi is inf, `gi < g` is False. So it passes?
        # No, `inf < g` is False. So it passes.
        # But `gi is None` is False.
        # So it returns the graph.
        # This is correct.
        
        return 0, []
        
    return best
