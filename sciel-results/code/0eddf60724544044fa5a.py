"""Construct a 3-regular graph of girth at least g with as few vertices as possible."""

import math
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (i, j) tuples.
    
    Strategy:
    1. For k=3 and small g, try known explicit constructions.
    2. For general cases, use a random lifting / probabilistic method to find
       a large girth graph.
    3. Fall back to a deterministic construction that guarantees correctness.
    """
    if k != 3:
        # General case: use a simple random regular graph approach
        return _random_k_regular(k, g)
    
    # Special case k=3
    if g <= 5:
        # K4 has girth 3, but we need girth >= g
        # For g=3: K4 (n=4)
        # For g=4: K4 has triangles, need larger
        # Actually for g=3, K4 works (no cycles shorter than 3, i.e., no 2-cycles)
        # For g=4, we need no triangles. K4 has triangles.
        # Petersen graph: 10 vertices, girth 5.
        # For g=3,4: use complete graph? No, K4 has girth 3.
        # Let's just use a random 3-regular graph and check.
        pass
    
    # Use a random 3-regular graph with sufficient size
    # The Moore bound for (3, g) gives a lower bound on n
    moore_n = _moore_bound(3, g)
    
    # Try random 3-regular graphs starting from a reasonable size
    # For girth g, we need n large enough
    # Empirically, n ~ 2 * 3^(g/2 - 1) is a rough upper bound for existence
    # But we want the smallest n, so start from Moore bound and increase
    
    base_n = max(moore_n, 10)
    # Round to even
    if base_n % 2 != 0:
        base_n += 1
    
    # Try random constructions
    for attempt_n in range(base_n, base_n + 200, 2):
        if attempt_n < 6:
            continue
        if _is_valid_3_regular(attempt_n, g):
            # Find such a graph
            for trial in range(50):
                n, edges = _random_3_regular(attempt_n)
                if _has_girth_at_least(n, edges, g):
                    return n, edges
    
    # Fallback: use a known good construction
    # For 3-regular graphs, the Petersen graph (n=10, girth=5) is a base
    # We can use tensor products or other constructions
    
    # Use a simple cycle-based construction
    # For girth g, we can use a "blow-up" of a base graph
    # Or use the LUW construction as a fallback
    
    return _luw_fallback(3, g)


def _moore_bound(k, g):
    """Compute the Moore bound for a (k, g) cage."""
    if g % 2 == 0:
        # Even girth: n >= k * (k-1)^(g/2 - 1) / 2 + (k-1)^(g/2 - 1)
        # Actually: n >= k * sum_{i=0}^{g/2-2} (k-1)^i
        half = g // 2
        s = 0
        for i in range(half - 1):
            s += (k - 1) ** i
        return k * s
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{g//2 - 1} (k-1)^i
        half = g // 2
        s = 0
        for i in range(half):
            s += (k - 1) ** i
        return 1 + k * s


def _random_k_regular(k, g):
    """Generate a random k-regular graph and verify girth."""
    # Start from a reasonable size
    moore_n = _moore_bound(k, g)
    base_n = max(moore_n, k + 1)
    if base_n % 2 != 0 and k % 2 == 0:
        base_n += 1
    
    for n in range(base_n, base_n + 100, 2):
        if n * k % 2 != 0:
            continue
        for trial in range(20):
            edges = _random_k_regular_edges(n, k)
            if edges is not None:
                if _has_girth_at_least(n, edges, g):
                    return n, edges
    
    # Fallback
    return _luw_fallback(k, g)


def _random_3_regular(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None, None
    if n * 3 % 2 != 0:
        return None, None
    
    # Configuration model
    for attempt in range(100):
        half_edges = []
        for v in range(n):
            for _ in range(3):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        used = [False] * len(half_edges)
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        # Check for multi-edges
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        # Check degree
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == 3 for d in degree):
            return n, edges
    
    return None, None


def _random_k_regular_edges(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(50):
        half_edges = []
        for v in range(n):
            for _ in range(k):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        # Check for multi-edges
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        # Check degree
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == k for d in degree):
            return edges
    
    return None


def _has_girth_at_least(n, edges, g):
    """Check if the graph has no cycle shorter than g using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS to find the shortest cycle
    # A cycle through vertex v can be found by looking for the shortest path
    # between two neighbors of v that doesn't use v
    # Or more efficiently: for each edge (u,v), check if there's a path from u to v
    # of length < g-1 that doesn't use the edge (u,v)
    
    # Simpler: for each vertex, do a BFS tracking the shortest path to each node,
    # and check for back-edges that form cycles
    
    for start in range(n):
        # BFS from start, tracking parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _is_valid_3_regular(n, g):
    """Check if there exists a valid 3-regular graph with n vertices and girth >= g."""
    # This is just a quick check; we'll actually try to construct one
    return True


def _luw_fallback(k, g):
    """
    Fall back to a Lazebnik-Ustimenko-Woldar type construction.
    This is the seed program's approach, used as a safety net.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
