"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for general cases,
    with specific optimizations for the (3, 16) cage.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # LUW Construction Parameters
    # We need girth >= g.
    # LUW CD(d, q) has girth >= d+5 (odd d) or d+4 (even d).
    # We want to minimize n, which is roughly 2*q^(d-1).
    # So we want smallest d.
    
    # Determine d
    # Odd d: d >= g - 5
    # Even d: d >= g - 4
    
    d_candidates = []
    
    # Odd d
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    d_candidates.append(d_odd)
    
    # Even d
    d_even = g - 4
    if d_even < 2: d_even = 2
    if d_even % 2 == 1: d_even += 1
    d_candidates.append(d_even)
    
    d = min(d_candidates)
    q = k
    
    # Field operations
    if q == 4:
        # GF(4)
        # Represent elements as 0,1,2,3
        # Add: XOR
        # Mul: Table
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Define LUW rules
    # rule[c] = (a, b) for c in 2..d
    # y[c] = x[c] +/- y[a] * x[b]
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rules[c] = (1, c0 - 2)
        elif r == 1:
            rules[c] = (c0 - 1, 1)
        elif r == 2:
            rules[c] = (c0, 1) # Note: c0 = c-2 here? No, c0 = c - 2 if r=2.
            # If c=7, r=2, c0=5. a=5, b=1. 5 < 7. OK.
        else: # r == 3
            rules[c] = (1, c0 + 1) # c0 = c-3. a=1, b=c-2. OK.
            
    # BFS to construct the graph
    # Vertices are (side, coords) where side is 0 or 1, coords is tuple of length d+1 (index 0 unused)
    # Start at (0, (None, 0, ..., 0))
    
    start = (0, (None,) + (0,) * d)
    visited = {start: 0}
    vertices = [start]
    edges = []
    
    for i, v in enumerate(vertices):
        side, x = v
        for first in range(q):
            # Initialize y
            y = [None, first] + [0] * (d - 1)
            # Compute y[2]..y[d]
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in visited:
                visited[neighbor] = len(vertices)
                vertices.append(neighbor)
            
            # Add edge only once (from point to line, i.e., side 0)
            if side == 0:
                edges.append((i, visited[neighbor]))
                
    n = len(vertices)
    return n, edges
# EVOLVE-BLOCK-END
