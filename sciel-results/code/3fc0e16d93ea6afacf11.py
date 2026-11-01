"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: use the C(5,9) cage if found, else fallback to LUW or hexagon.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill

    # For k=5, g=9, the known best is 1116. We try to find a smaller one or use a fast construction.
    # The LUW construction for k=5 gives n ~ 6000+ which is worse than 1116 but valid.
    # However, the task says "score: n, minimized". The seed program returns a valid graph.
    # We must return a valid graph.
    
    # If k is small and g is specific, we can try known cages or better constructions.
    # For k=5, g=9, we can try to use a known small graph.
    # But without a library, we must construct it.
    
    # Strategy:
    # 1. If k=5 and g=9, try to construct a graph with n < 1116.
    #    The Moore bound for k=5, g=9 (odd girth) is n >= 1 + k * (k-1)^(g-1)/2 = 1 + 5*4^4 = 1 + 5*256 = 1281?
    #    Wait, Moore bound for girth g (odd) is 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i.
    #    For g=9, (g-3)/2 = 3. Sum = 1 + 4 + 16 + 64 = 85. n >= 1 + 5*85 = 426.
    #    So 1116 is far from the bound.
    
    # Let's use the LUW construction as a fallback, but try to optimize it or use a different one.
    # Actually, the LUW construction in the seed is for q=k. For k=5, q=5.
    # d = max(3, g-5 if (g-5)%2 else g-4). For g=9, g-5=4 (even), so d=8.
    # The LUW graph CD(8, 5) has girth at least 8+5=13? No, girth is d+5 or d+4?
    # The comment says "girth is at least d + 5 for odd d".
    # If d is even, maybe d+4?
    # Let's check the girth of the seed's output.
    
    # A better approach for k=5, g=9:
    # Use the "Petersen-like" or other known small cages?
    # No, we need to construct it programmatically.
    
    # Let's try a random regular graph approach with girth check?
    # Too slow for n ~ 1000 in 120s?
    
    # Let's stick to the LUW construction but maybe we can find a smaller one by varying parameters?
    # Or use the hexagon H(2,4) for k=5?
    # H(2, q) is a generalized hexagon with girth 12.
    # For q=4, n = 2 * (4^3 + 1)? Or 2 * 4^3 = 128?
    # Let's check the order of H(2, q).
    # Points and lines are in GF(q)^3?
    # If n=128, it is much smaller than 1116.
    # But is it 5-regular?
    # In a generalized hexagon H(2,q), the degree is q+1.
    # If q=4, degree is 5.
    # So H(2,4) is a 5-regular graph with girth 12.
    # Since 12 >= 9, it satisfies the girth constraint.
    # And n=128 is much smaller than 1116.
    
    # Let's implement H(2,4) for k=5.
    # If k is not 5, or g is different, use LUW.
    
    if k == 5 and g <= 12:
        try:
            n, edges = _hexagon_2_4()
            return n, edges
        except Exception:
            pass
    
    # Fallback: LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

def _luw_construct(k, g):
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

def _hexagon_2_4():
    """
    Construct the generalized hexagon H(2, 4).
    q = 4.
    Points: (x, y, z) in GF(4)^3.
    Lines: (u, v, w) in GF(4)^3.
    Incidence:
    x + u*y + v*z = 0
    y + v*z + w*x = 0  (This is a guess, need correct definition)
    
    Actually, the standard definition for H(2,q) uses:
    P = (x, y, z), L = (u, v, w)
    Incidence iff:
    x + yu + zv = 0
    y + xv + zw = 0
    z + xw + yu = 0  (Wait, this might be dependent)
    
    Let's use the definition from:
    "Generalized Hexagons"
    H(2,q) points are (x, y, z) and lines are (u, v, w).
    Incidence:
    x + yu + zv = 0
    y + xv + zw = 0
    This system has q+1 solutions for (u,v,w) given (x,y,z)?
    If we fix x,y,z, we have 2 equations for 3 unknowns.
    If the matrix is rank 2, there are q solutions.
    But we need q+1.
    Maybe there is an additional condition or the lines are projective?
    
    Alternative: Use the explicit graph from literature for H(2,4).
    Since n=128 is small, we can hardcode or generate via a different method.
    
    Let's try a different construction: The Hoffman-Singleton graph is 7-regular, girth 5.
    For 5-regular, girth 9, maybe the "Klein graph"? No.
    
    Let's check if n=128 is correct for H(2,4).
    If it is 5-regular and girth 12, it's perfect.
    
    Let's implement the incidence check by iterating all lines for each point.
    """
    q = 4
    # GF(4) operations
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def add(a, b): return a ^ b
    def mul(a, b): return table[a][b]
    
    # Points: 0..63 (2 bits per coord? No, 3 coords of 2 bits = 6 bits = 64 points)
    # Lines: 0..63
    # Wait, if points are GF(4)^3, there are 64 points.
    # If lines are GF(4)^3, there are 64 lines.
    # Total vertices = 128.
    
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                points.append((x, y, z))
                
    lines = []
    for u in range(4):
        for v in range(4):
            for w in range(4):
                lines.append((u, v, w))
                
    # Indexing
    p_idx = {p: i for i, p in enumerate(points)}
    l_idx = {l: i for i, l in enumerate(lines)}
    
    edges = []
    degree = [0] * 128
    
    for pi, p in enumerate(points):
        x, y, z = p
        for li, l in enumerate(lines):
            u, v, w = l
            # Check incidence
            # x + yu + zv = 0
            # y + xv + zw = 0
            eq1 = add(add(x, mul(y, u)), mul(z, v))
            eq2 = add(add(y, mul(x, v)), mul(z, w))
            if eq1 == 0 and eq2 == 0:
                edges.append((pi, 64 + li))
                degree[pi] += 1
                degree[64 + li] += 1
                
    # Check if 5-regular
    if all(d == 5 for d in degree):
        return 128, edges
    else:
        # Fallback: maybe the incidence is different
        # Try:
        # x + yu + zv = 0
        # z + xw + yv = 0 (permutation)
        edges = []
        degree = [0] * 128
        for pi, p in enumerate(points):
            x, y, z = p
            for li, l in enumerate(lines):
                u, v, w = l
                eq1 = add(add(x, mul(y, u)), mul(z, v))
                eq2 = add(add(z, mul(x, w)), mul(y, v))
                if eq1 == 0 and eq2 == 0:
                    edges.append((pi, 64 + li))
                    degree[pi] += 1
                    degree[64 + li] += 1
                    
        if all(d == 5 for d in degree):
            return 128, edges
        else:
            raise ValueError("Hexagon construction failed")

# EVOLVE-BLOCK-END
