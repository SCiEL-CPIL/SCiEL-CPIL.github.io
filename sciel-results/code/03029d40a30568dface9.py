import time
import sys
from collections import deque

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320.
    # We search for a circulant graph C(n, {s1, s2})
    
    # Pre-compute LUW result as a fallback
    luw_result = None
    
    # Search for circulant graphs
    # Try n from Moore bound up to a reasonable limit
    # Prioritize n that are even (for bipartite structure)
    
    # First pass: try even n values (more likely to be bipartite)
    for n in range(246, 600, 2):
        if time.time() - t_start > _DEADLINE - 10:
            break
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 10:
                break
            for s2 in range(s1 + 1, max_step + 1):
                # Quick 4-cycle check: s1+s2 != 0 mod n
                if (s1 + s2) % n == 0:
                    continue
                # Check girth using optimized method
                if _check_girth_circulant_fast(n, s1, s2, 10):
                    edges = _build_circulant_edges(n, s1, s2)
                    return n, edges
    
    # Second pass: try odd n values if time permits
    for n in range(247, 600, 2):
        if time.time() - t_start > _DEADLINE - 10:
            break
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 10:
                break
            for s2 in range(s1 + 1, max_step + 1):
                if (s1 + s2) % n == 0:
                    continue
                if _check_girth_circulant_fast(n, s1, s2, 10):
                    edges = _build_circulant_edges(n, s1, s2)
                    return n, edges
    
    # If no circulant found, fallback to LUW
    return _luw(k, g)

def _build_circulant_edges(n, s1, s2):
    """Build edge list for circulant graph C(n, {s1, s2})."""
    edges = []
    for u in range(n):
        v1 = (u + s1) % n
        v2 = (u + s2) % n
        if u < v1:
            edges.append((u, v1))
        if u < v2:
            edges.append((u, v2))
    return edges

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    Uses BFS with early termination.
    """
    # For girth >= g, we need no cycles of length < g
    # BFS from 0, track distances
    # If we find an edge (u,v) where dist[u] + dist[v] + 1 < g and dist[u] != dist[v] + 1 and dist[v] != dist[u] + 1, we have a short cycle
    
    max_depth = g // 2 - 1  # For g=10, max_depth=4
    
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= max_depth:
            continue
        
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if dist[v] == -1:
                dist[v] = d + 1
                q.append(v)
            else:
                # Check if this edge creates a short cycle
                # Cycle length = dist[u] + dist[v] + 1
                # But we need to ensure it's a simple cycle (no backtracking)
                # If |dist[u] - dist[v]| <= 1, it's a tree edge or cross edge
                # A cycle exists if dist[u] + dist[v] + 1 < g
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    
    # Also check that the BFS tree has the expected number of nodes at each level
    # For a graph with girth >= g, the number of nodes at distance d from 0 is:
    # d=0: 1
    # d=1: k=4
    # d=2: k*(k-1)=12
    # d=3: k*(k-1)^2=36
    # d=4: k*(k-1)^3=108
    if n < 1 + 4 + 12 + 36 + 108:  # 161
        return False
    
    # Count nodes at each distance
    counts = [0] * (max_depth + 1)
    for d in range(n):
        if dist[d] >= 0 and dist[d] <= max_depth:
            counts[dist[d]] += 1
    
    expected = [1, 4, 12, 36, 108]
    for d in range(1, max_depth + 1):
        if d < len(expected) and counts[d] < expected[d]:
            return False
    
    return True

def _check_girth_circulant(n, s1, s2, g):
    """Legacy girth check (kept for compatibility)."""
    return _check_girth_circulant_fast(n, s1, s2, g)

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
