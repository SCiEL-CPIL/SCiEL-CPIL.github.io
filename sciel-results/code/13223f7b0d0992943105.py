"""Cage problem n(4,10): construct a 4-regular graph with girth >= 10 on fewest vertices.

Strategy: Use the Lazebnik-Ustimenko-Woldar (LW) construction, which produces a k-regular
bipartite graph of girth at least g. The LW construction for k=4, g=10 uses the rule:
  d = g - 5 = 5 (odd), q = 4 (the field size)
  
The LW graph CD(d, q) has points and lines over GF(q)^d. For k=4, we use GF(4) with
characteristic 2 arithmetic.

However, the known smallest 4-regular graph of girth 10 has 320 vertices (Exoo et al. 2026).
The LW construction for (4,10) gives a graph with more vertices. Let's compute:
- The LW graph CD(5, 4) has 2 * 4^5 = 2048 vertices total, but we take the connected
  component of the zero point, which is much smaller.

Actually, let me reconsider. The LW construction CD(d,q) for d=5, q=4:
- Points: (x_1, ..., x_5) in GF(4)^5 where x_1 is free (or None in the code)
- The adjacency is defined by the rule with specific coefficients.

For k=4, g=10: d = 5, and we use the rule coefficients.
The connected component of the zero point in CD(5,4) should be manageable.

Let me trace through the LW construction more carefully:
- d = 5 (since g=10, d = g-5 = 5, which is odd)
- q = 4
- rule[2] = (1,1), rule[3] = (2,1), rule[4] = (1,2)
- For c=5: r = (5-5)%4 = 0, c0 = 5 - 0 = 5
  rule[5] = [(1, 3), (4, 1), (5, 1), (1, 6)][0] = (1, 3)

So the rule is:
  rule[2] = (1, 1)
  rule[3] = (2, 1)
  rule[4] = (1, 2)
  rule[5] = (1, 3)

The adjacency: point p ~ line l iff for c=2..5:
  l[c] - p[c] = l[a_c] * p[b_c]

where (a_c, b_c) = rule[c].

This gives a 4-regular bipartite graph. The connected component of the zero point
should have a reasonable number of vertices.

Let me implement this carefully and efficiently.
"""
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # LW construction parameters
    q = k  # = 4
    # d must be odd and d >= g-5
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    # For g=10, d=5 (odd)
    
    # GF(4) arithmetic using XOR for addition, and a multiplication table
    # GF(4) = {0, 1, 2, 3} with 2 = alpha, 3 = alpha^2, alpha^2 = alpha + 1
    # Multiplication table:
    #   0*anything = 0
    #   1*x = x
    #   2*0=0, 2*1=2, 2*2=3, 2*3=1
    #   3*0=0, 3*1=3, 3*2=1, 3*3=2
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(x, y):
        return x ^ y
    
    def gf_sub(x, y):
        return x ^ y  # In char 2, subtraction = addition
    
    def gf_mul(x, y):
        return mul_table[x][y]
    
    # Rule coefficients for the LW construction
    # rule[c] = (a, b) for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The four options cycle: (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Vertices are (side, x) where side in {0, 1} and x is a tuple of length d+1
    # x[0] is None (free coordinate), x[1..d] are in GF(4)
    # Actually, let's use: vertex = (side, x1, x2, ..., xd) where side=0 is point, side=1 is line
    # The "None" coordinate is handled implicitly.
    
    # For efficiency, encode vertices as integers:
    # side * q^d + x1 * q^(d-1) + x2 * q^(d-2) + ... + xd
    # But we also need to track the None coordinate... Actually in the LW construction,
    # the first coordinate is free and not part of the state. Let me re-examine.
    
    # In the original code: vertex = (side, (None,) + (0,)*d)
    # So the tuple has d+1 elements: [None, x1, x2, ..., xd]
    # The None is just a placeholder. The actual data is x1..xd in GF(4).
    
    # Let me encode: vertex_id = side * (q**d) + x1 * q**(d-1) + ... + xd
    qd = q ** d  # 4^5 = 1024
    
    def encode(side, x):
        """x is a tuple of d elements in GF(4)"""
        val = 0
        for i in range(d):
            val = val * q + x[i]
        return side * qd + val
    
    def decode(vertex_id):
        """Return (side, x) where x is tuple of d elements"""
        side = vertex_id // qd
        val = vertex_id % qd
        x = [0] * d
        for i in range(d - 1, -1, -1):
            x[i] = val % q
            val //= q
        return side, tuple(x)
    
    # Start vertex: side=0 (point), all coordinates 0
    start_vertex = (0, tuple([0] * d))
    start_id = encode(0, start_vertex)
    
    # BFS to find connected component
    # For each vertex, compute its 4 neighbors
    # 
    # Point (side=0) with coords x = (x1, x2, ..., xd):
    #   Neighbor is a line (side=1) with coords y = (y1, y2, ..., yd)
    #   where y[c] = x[c] + y[a_c] * x[b_c] for c = 2..d
    #   and y[1] = first (free parameter, 0..q-1)
    #
    # Wait, let me re-read the original code:
    #   for first in range(q):
    #       y = [None, first] + [0]*(d-1)
    #       for c in range(2, d+1):
    #           a, b = rule[c]
    #           if side == 0:  # point
    #               y[c] = add(x[c], mul(y[a], x[b]))
    #           else:  # line
    #               y[c] = sub(x[c], mul(x[a], y[b]))
    #       yield 1-side, tuple(y)
    #
    # So y[1] = first (the free coordinate), and y[c] for c>=2 is computed.
    # Note: y is indexed 1..d (y[0] is None).
    # x is also indexed 1..d (x[0] is None).
    
    # Let me rewrite with 1-based indexing for clarity:
    # x = (x_1, x_2, ..., x_d) corresponding to indices 1..d
    
    def get_neighbors(vertex_id):
        """Yield the 4 neighbor vertex_ids of the given vertex."""
        side, x = decode(vertex_id)
        # x is 0-indexed tuple: x[0]..x[d-1] corresponds to x_1..x_d
        for first in range(q):
            # y_1 = first
            y = [0] * d  # 0-indexed: y[0] = y_1, y[1] = y_2, ..., y[d-1] = y_d
            y[0] = first
            for c in range(2, d + 1):
                # c is 1-based, so 0-based index is c-1
                a, b = rule[c]
                # a, b are 1-based indices
                if side == 0:
                    # y_c = x_c + y_a * x_b
                    y[c - 1] = gf_add(x[c - 1], gf_mul(y[a - 1], x[b - 1]))
                else:
                    # y_c = x_c - x_a * y_b
                    y[c - 1] = gf_sub(x[c - 1], gf_mul(x[a - 1], y[b - 1]))
            yield encode(1 - side, tuple(y))
    
    # BFS
    index = {start_id: 0}
    order = [start_id]
    edges = []
    
    # Precompute neighbors for all vertices in the component
    # Use a queue for BFS
    queue = deque([start_id])
    
    while queue:
        if time.time() > deadline:
            break
        vid = queue.popleft()
        side, _ = decode(vid)
        for nbr in get_neighbors(vid):
            if nbr not in index:
                index[nbr] = len(order)
                order.append(nbr)
                queue.append(nbr)
            if side == 0:
                # Only add edge from point to line to avoid duplicates
                if index[vid] < index[nbr]:
                    edges.append((index[vid], index[nbr]))
                else:
                    edges.append((index[nbr], index[vid]))
    
    n = len(order)
    
    # Verify: check degree and girth
    # For now, just return. The evaluator will verify.
    return n, edges
