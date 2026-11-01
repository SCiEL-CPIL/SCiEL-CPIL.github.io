import time
import sys
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Try to find a circulant graph C_320(a, b) with girth >= 10
        n_circ = 320
        found = _find_circulant(n_circ, k, g, deadline)
        if found:
            return n_circ, found
        
        # Fallback to LPS q=13 (n=2184) if circulant fails or not found
        n, edges = _try_lps(13)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges
            
        # Fallback to LPS q=11 (n=1320) - check if girth is sufficient
        n, edges = _try_lps(11)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges

        # Fallback to LPS q=7 (n=336) - likely fails girth 10
        n, edges = _try_lps(7)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges

    # Fallback: LUW construction for general k, g
    return _luw_construct(k, g)

def _find_circulant(n, k, g, deadline):
    """Search for a k-regular circulant graph on n vertices with girth >= g."""
    # For k=4, generators are +/-a, +/-b
    # We need a, b in 1..n//2-1
    # Check constraints for no small cycles
    for a in range(1, n // 2):
        if time.time() > deadline:
            return None
        for b in range(a + 1, n // 2):
            # Check for triangles/squares
            # 2a = 0, 2b = 0, a+b=0, a-b=0
            if (2 * a) % n == 0 or (2 * b) % n == 0 or (a + b) % n == 0 or (a - b) % n == 0:
                continue
            
            # Check girth
            if _check_circulant_girth(n, a, b, g):
                # Construct edges
                edges = []
                for i in range(n):
                    for gen in [a, -a, b, -b]:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
                return edges
    return None

def _check_circulant_girth(n, a, b, g):
    """Check if C_n(a, b) has girth >= g."""
    # BFS from 0
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        v = queue.popleft()
        if dist[v] >= g // 2:
            continue
        for gen in [a, -a, b, -b]:
            w = (v + gen) % n
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                queue.append(w)
            else:
                # If we reach a visited node, check cycle length
                # Cycle length is dist[v] + 1 + dist[w] ? No.
                # If w is visited, the edge (v, w) closes a cycle.
                # The cycle length is dist[v] + 1 + (dist[w] if w is not parent?)
                # Actually, in a BFS tree, if we find an edge (v, w) where w is already visited,
                # the cycle length is dist[v] + 1 + dist[w] only if the paths are disjoint?
                # No, the cycle is 0 -> ... -> v -> w -> ... -> 0.
                # Length is dist[v] + 1 + dist[w].
                # We need this to be >= g.
                # Also, if w is the parent of v, it's a back edge, not a new cycle.
                # But in a circulant graph, we don't track parents easily in this simple dist array.
                # However, if dist[w] == dist[v] + 1, it's a forward edge (tree edge).
                # If dist[w] == dist[v], it's a cross edge. Cycle length = 2 * dist[v] + 1?
                # If dist[w] < dist[v], it's a back edge. Cycle length = dist[v] - dist[w] + 1.
                
                # Simpler: if dist[v] + dist[w] + 1 < g, return False.
                # This is a sufficient condition for a short cycle.
                if dist[v] + dist[w] + 1 < g:
                    return False
                    
    return True

def _try_lps(q):
    # ... (same as before) ...
    pass

def _check_girth(n, edges, g, deadline):
    # ... (same as before) ...
    pass

def _luw_construct(k, g):
    # ... (same as before) ...
    pass
