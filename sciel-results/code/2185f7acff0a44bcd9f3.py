"""
Construct small k-regular graphs of given girth.

The seed program uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
For k=3, g=16, the LUW construction with d=11, q=3 gives a 3-regular
bipartite graph with girth >= 16. The number of vertices is 2 * 3^10 = 8856,
which is far above the known best of 936.

This is a research program for the cage problem. The task asks us to
construct a 3-regular graph with girth at least 16 on as few vertices as
possible. The known best is 936 vertices.

Since we cannot easily construct the 936-vertex cage algorithmically in
this context, and the LUW construction gives a much larger graph, we need
a different approach.

One approach: use the fact that for small k and g, we can use known
constructions or search methods. However, given the constraints, let's
try to use a different explicit construction.

Actually, let me reconsider. The task says "n must lie between the Moore
bound and max_vertices". The Moore bound for (3, 16) is:
  2 * (1 + 2 + 2^2 + ... + 2^7) = 2 * (2^8 - 1) = 510

So n must be >= 510. The known best is 936.

Let me try a different approach: use a greedy construction or a known
small cage if we can encode it.

Actually, for the purpose of this exercise, let me implement a more
general approach that tries to construct the graph using a combination of:
1. LUW construction when applicable
2. For specific small cases, use known constructions

For k=3, g=16, the known best construction is due to Exoo et al. (2026)
with 936 vertices. We cannot easily reproduce this.

Let me try a different explicit construction: the "cage" can sometimes be
constructed using algebraic methods. For (3, 16), one approach is to use
the incidence graph of a certain design or a Cayley graph.

Actually, let me just improve the LUW construction to be more efficient
and see if we can get a smaller graph by trying different parameters.

Wait - the LUW construction CD(d, q) gives a q-regular bipartite graph.
For q=3, the number of vertices in the connected component containing
the origin is at most 2 * 3^(d-1).

For d=11: 2 * 3^10 = 8856
For d=10 (even): girth >= 14, not enough for g=16
For d=11 (odd): girth >= 16, n <= 8856

But the actual connected component might be smaller. Let me check.

Actually, I think the issue is that the LUW construction might not give
the full 2*q^(d-1) vertices in the connected component. Let me just
implement it correctly and see what n we get.

Let me rewrite the program with a cleaner implementation and also add
a fallback for small cases where we know the answer.
"""
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Returns (n, edges) where n is the number of vertices and edges is
    a list of (i, j) pairs.
    """
    if k == 1:
        return 2, [(0, 1)]
    
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    # For k >= 3, use LUW construction
    # CD(d, q): q-regular bipartite graph
    # Girth >= d+5 if d odd, >= d+4 if d even
    
    # Find minimal d such that girth >= g
    # If d odd: d + 5 >= g => d >= g - 5
    # If d even: d + 4 >= g => d >= g - 4
    
    d_candidates = []
    
    # Try odd d
    d_odd = g - 5
    if d_odd < 3:
        d_odd = 3
    if d_odd % 2 == 0:
        d_odd += 1
    d_candidates.append(d_odd)
    
    # Try even d
    d_even = g - 4
    if d_even < 2:
        d_even = 2
    if d_even % 2 == 1:
        d_even += 1
    d_candidates.append(d_even)
    
    d = min(d_candidates)
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # LUW rule
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
    
    # BFS to find connected component
    # Vertex represented as (side, tuple of d values)
    start = (0, (0,) * d)
    visited = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [0] * d
            y[0] = first
            for c in range(1, d):
                a, b = rule[c + 1]
                if side == 0:
                    y[c] = add(x[c], mul(y[a - 1], x[b - 1]))
                else:
                    y[c] = sub(x[c], mul(x[a - 1], y[b - 1]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in visited:
                visited[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((i, visited[neighbor]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
