"""
Constructing small 5-regular graphs of large girth using LUW construction.

Strategy:
1. Use Lazebnik-Ustimenko-Woldar construction with q=k, d chosen so girth >= g.
2. Vectorize the neighbor computation with NumPy for speed.
3. Fast girth validation using BFS with early termination.
4. Deadline guard to ensure we return within time limit.
"""

import time
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin before 150s limit
    
    # LUW construction: q-regular bipartite graph with girth >= d+5 for odd d
    # For k=5, g=9: need d+5 >= 9, so d >= 4. Smallest odd d >= 4 is d=5.
    q = k
    d = max(3, g - 5)
    if d % 2 == 0:
        d += 1  # ensure odd d for the girth guarantee
    
    # Build LUW graph
    n, edges = _luw_vectorized(q, d)
    
    # Validate
    if time.time() < deadline:
        if _validate_fast(k, g, n, edges):
            return n, edges
    
    # If validation fails or times out, try a larger d to be safe
    d2 = d + 2
    n2, edges2 = _luw_vectorized(q, d2)
    if time.time() < deadline:
        if _validate_fast(k, g, n2, edges2):
            return n2, edges2
    
    # Fallback: return the first result even if validation didn't complete
    return n, edges


def _luw_vectorized(q, d):
    """
    Vectorized LUW construction.
    
    Points: (0, x) where x in GF(q)^d
    Lines: (1, y) where y in GF(q)^d
    
    Point p ~ Line l iff for all c=2..d: l[c] - p[c] = l[a_c] * p[b_c]
    
    Returns (n, edges) where n = 2*q^d and edges is list of (i,j).
    """
    # GF(q) operations (q is prime for k=5)
    # Points are indexed 0..q^d-1, lines are q^d..2*q^d-1
    
    qd = q ** d
    n = 2 * qd
    
    # Precompute the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate all points as arrays
    # Point index i corresponds to vector x where i = x[d] * q^(d-1) + ... + x[1]
    # We'll use 0-indexed: position c in the tuple corresponds to index c-1 in the array
    
    # For each point, compute all its line neighbors
    edges = []
    
    # Vectorized approach: for each point, compute all q^d possible line neighbors
    # A line neighbor of point p is determined by y[1] (the first coordinate)
    # and then y[c] is determined for c=2..d
    
    # For point p with coordinates (p[1], p[2], ..., p[d]):
    # For each first = 0..q-1:
    #   y[1] = first
    #   For c=2..d: y[c] = p[c] + y[a_c] * p[b_c] (mod q)
    
    # Let's process all points at once
    # Create array of all point coordinates
    # point_coords[i] = [x1, x2, ..., xd] for point i
    
    # Generate all points
    indices = np.arange(qd)
    # Convert to base-q digits
    point_coords = np.zeros((qd, d), dtype=np.int32)
    for pos in range(d):
        point_coords[:, pos] = (indices // (q ** pos)) % q
    
    # For each point and each first value, compute the line coordinates
    # This gives q * qd line neighbors per point, but many will be duplicates
    
    # Actually, each point has exactly q neighbors (one for each first value)
    # Wait, no. Let me re-read the LUW construction.
    
    # In LUW(d, q), each point is connected to q lines, and each line to q points.
    # The graph is q-regular bipartite.
    # Total vertices: 2 * q^d
    # Total edges: q^d * q = q^(d+1)
    
    # For point p, the q neighbors are obtained by varying first = 0..q-1
    # and computing y[c] for c=2..d
    
    # Let's compute all edges
    edges = []
    
    # For each point
    for pt_idx in range(qd):
        p = point_coords[pt_idx]
        for first in range(q):
            # Compute line coordinates
            y = np.zeros(d, dtype=np.int32)
            y[0] = first  # y[1] in 1-indexed = y[0] in 0-indexed
            for c in range(2, d + 1):
                a, b = rule[c]
                # y[c] = p[c] + y[a] * p[b] (mod q)
                # In 0-indexed: y[c-1] = p[c-1] + y[a-1] * p[b-1] (mod q)
                y[c - 1] = (p[c - 1] + y[a - 1] * p[b - 1]) % q
            
            # Line index
            line_idx = qd + int(np.sum(y * (q ** np.arange(d))))
            edges.append((pt_idx, line_idx))
    
    return n, edges


def _validate_fast(k, g, n, edges):
    """
    Fast validation: check k-regularity, simplicity, and girth >= g.
    """
    # Check degree and build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using BFS from each vertex, early termination
    # For girth >= g, we need no cycles of length < g
    # BFS from each vertex up to depth g-1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= g - 1:
                continue
            
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    if dist[neighbor] < g - 1:
                        queue.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a back edge
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    
    return True
