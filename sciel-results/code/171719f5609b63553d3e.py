"""Rewritten program: explicit high-girth constructions + verification with deadline guard."""
import time
import sys
import numpy as np
from collections import deque

# Hard wall-clock limit (evaluator kills at 150s; keep safety margin)
_DEADLINE = 150.0
_T0 = None


def _elapsed():
    return time.time() - _T0


def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin


def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g on as few vertices as possible."""
    global _T0
    _T0 = time.time()

    # Strategy:
    # 1. Try known explicit constructions (Lazebnik-Ustimenko-Woldar, etc.)
    # 2. Verify girth via non-backtracking BFS
    # 3. If time permits, try to find smaller via other constructions
    # Return the best valid result found.

    best = None  # (n, edges)

    # --- Construction 1: Lazebnik-Ustimenko-Woldar (LUW) CD(d, q) ---
    # For k = q (prime or 4), girth >= d+5 (d odd) or d+4 (d even)
    # We need girth >= g, so choose d appropriately.
    # The graph is bipartite with q*d vertices per side (approximately), total ~ 2*q*d.
    # Actually the connected component of zero may be smaller.

    if k in (2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        # Try LUW
        result = _luw_construct(k, g)
        if result is not None:
            best = result

    # --- Construction 2: For k=5, try known small cages or algebraic constructions ---
    # For 5-regular girth 9, the known best is 1116 vertices.
    # LUW with q=5, d=4 gives girth >= 4+5=9? Let's check: d=4 even -> girth >= d+4 = 8? 
    # Actually the LUW construction: for d odd, girth >= 2d+1... let me re-derive.
    # The standard LUW: CD(d,q) has girth >= 2d+1 if d is odd, >= 2d+2 if d is even? 
    # No: the paper states girth >= d+5 for odd d, d+4 for even d.
    # So for g=9: need d+5 >= 9 -> d >= 4 (odd: d=5 gives girth>=10, d=3 gives girth>=8)
    # For even d: d+4 >= 9 -> d >= 5, but d even so d=6 gives girth>=10.
    # Let's just try multiple d values.

    # Actually let me be more careful. The LUW graph CD(d,q):
    # Vertices: points and lines in PG(d-1, q) roughly.
    # The girth is at least 2*min(d, q-1) + 1 or similar.
    # For our purposes, let's just build it and verify.

    # --- Construction 3: Petersen-type or other small graphs for small k,g ---
    # For k=2: cycle C_g, n=g
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        # Verify
        if _verify(n, edges, k, g):
            best = (n, edges)
        return best

    # For k=3, g=4: K4 (n=4)
    # For k=3, g=5: Petersen (n=10)
    # For k=3, g=6: Heawood (n=14)
    # For k=3, g=7: (3,7)-cage is n=24 (Tutte-Coxeter)
    # For k=3, g=8: (3,8)-cage is n=30
    # For k=3, g=9: (3,9)-cage is n=34 (Foster 153? No, it's the (3,9)-cage with 34 vertices)
    # Actually (3,9)-cage: n=34 is known.

    if k == 3:
        # Use known small cages or LUW
        if g <= 4:
            n = 4
            edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
            if _verify(n, edges, k, g):
                best = (n, edges)
        elif g <= 5:
            n = 10
            # Petersen graph
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            if _verify(n, edges, k, g):
                best = (n, edges)
        elif g <= 6:
            n = 14
            # Heawood graph (incidence of Fano plane)
            edges = []
            # 7 points, 7 lines
            # Points 0-6, Lines 7-13
            # Fano plane lines:
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            if _verify(n, edges, k, g):
                best = (n, edges)
        elif g <= 7:
            n = 24
            # Tutte-Coxeter graph (3,7)-cage
            # This is the incidence graph of the projective plane of order 3? No, that's (4,6).
            # The (3,7)-cage is the Tutte-Coxeter graph with 24 vertices.
            # Let me construct it as the Levi graph of the (3,7)-config or use known adjacency.
            # Actually, the (3,7)-cage is the incidence graph of the projective plane PG(2,3)? 
            # PG(2,3) has 13 points and 13 lines, each line has 4 points. That gives (4,6)-cage? No.
            # Let me just use LUW for k=3.
            pass  # Fall through to LUW

    # For k=5, g=9: LUW with q=5
    # Let's build LUW for various parameters and pick the smallest valid one.

    if best is None:
        # Try LUW with different d values
        for d in range(3, 15):
            if not _time_ok(margin=10.0):
                break
            result = _luw_construct(k, g, d_override=d)
            if result is not None:
                if best is None or result[0] < best[0]:
                    best = result

    # If still nothing, try a simple random/regular graph construction (last resort)
    if best is None:
        # Use a circulant or other simple construction
        best = _simple_regular_graph(k, g)

    return best


def _luw_construct(k, g, d_override=None):
    """Build Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k."""
    q = k
    if q not in (4, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    # Choose d: need girth >= g
    # LUW girth: for odd d, >= d+5; for even d, >= d+4
    # So we need: if d odd, d >= g-5; if d even, d >= g-4
    # Pick smallest valid d
    if d_override is not None:
        d = d_override
    else:
        # Find smallest d such that girth bound >= g
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    # Build GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Define the rule for the CD(d,q) construction
    # The rule maps c -> (a, b) such that the incidence is defined
    # Standard LUW rule:
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Build the graph using BFS from the zero vertex
    # Vertex representation: (side, coords) where side 0 = point, side 1 = line
    # coords is a tuple of length d+1 (index 0 unused, indices 1..d)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point x, line y: y[c] = x[c] + y[a]*x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line x, point y: y[c] = x[c] - x[a]*y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    if n == 0:
        return None

    # Verify
    if _verify(n, edges, k, g):
        return (n, edges)
    return None


def _simple_regular_graph(k, g):
    """Fallback: construct a simple k-regular graph and hope girth is sufficient.
    This is a last resort and may not satisfy girth."""
    # Use a circulant graph: vertices 0..n-1, edges (i, i+1 mod n), (i, i+2 mod n), ...
    # For k even: connect i to i±1, i±2, ..., i±k/2
    # Girth of circulant C(n; 1,2,...,k/2) is approximately 2*floor(n/(k/2))+1 for small n
    # This is very rough.
    
    # Better: use a known construction. For now, just return a star-like or cycle-based graph.
    # Actually, let's try to build a random k-regular graph and check girth.
    # But this is complex. Let's just return the LUW result if it exists, otherwise a simple cycle.
    
    # For k=2, already handled.
    # For general k, let's try a simple approach:
    # Build a k-regular graph on n vertices where n is large enough.
    # Use the configuration model or just a circulant.
    
    n = 2 * k * (g - 1)  # Rough upper bound
    edges = []
    # Circulant: for each i, connect to i+1, i-1, i+2, i-2, ..., i+k//2, i-k//2
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % n))
    # Remove duplicates and self-loops
    seen = set()
    unique_edges = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            unique_edges.append(key)
    
    # Check if it's k-regular
    deg = [0] * n
    for u, v in unique_edges:
        deg[u] += 1
        deg[v] += 1
    if all(d == k for d in deg):
        if _verify(n, unique_edges, k, g):
            return (n, unique_edges)
    
    # If not valid, try larger n
    for factor in range(2, 10):
        n2 = n * factor
        edges2 = []
        for i in range(n2):
            for j in range(1, k // 2 + 1):
                edges2.append((i, (i + j) % n2))
        seen2 = set()
        unique_edges2 = []
        for u, v in edges2:
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key not in seen2:
                seen2.add(key)
                unique_edges2.append(key)
        deg2 = [0] * n2
        for u, v in unique_edges2:
            deg2[u] += 1
            deg2[v] += 1
        if all(d == k for d in deg2):
            if _verify(n2, unique_edges2, k, g):
                return (n2, unique_edges2)
    
    # Absolute fallback: return a trivial invalid result (shouldn't happen)
    return (1, [])


def _verify(n, edges, k, g):
    """Verify that the graph is k-regular, simple, and has girth >= g.
    Uses non-backtracking BFS from every vertex to certify girth."""
    if n == 0:
        return False
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check no duplicate edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check girth >= g using non-backtracking BFS
    # For each vertex, do BFS avoiding backtracking, track depth
    # If we find a cycle of length < g, return False
    # Non-backtracking walk: from vertex v, we can go to neighbor w, but not back to the vertex we came from
    
    for start in range(n):
        # BFS from start, tracking (current_vertex, previous_vertex, depth)
        # State: (v, prev) where prev is the vertex we came from (or -1 for start)
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                # We've reached depth g-1 without finding a cycle < g
                # But we need to check if we can continue
                continue
            for w in adj[v]:
                if w == prev:
                    continue  # Don't backtrack
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                # If w == start and depth + 1 >= 2, we found a cycle
                # But non-backtracking cycles: if w == start, the cycle length is depth + 1
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True
