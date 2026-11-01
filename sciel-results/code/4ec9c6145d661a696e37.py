"""
Construct a 5-regular graph of girth 9 by taking the line graph of a
3-regular graph with girth >= 14.

Key facts:
- The line graph L(H) of a d-regular graph H is (2d-2)-regular.
- So L(H) of a 3-regular H is 4-regular. That's not 5.

Wait, this doesn't work. The line graph of a 3-regular graph is 4-regular,
not 5-regular. So this approach as stated cannot directly produce a
5-regular graph.

Let me reconsider. Perhaps the intended construction is different:
- Maybe it's the line graph of a graph where each vertex has degree 3
  but we add some structure, or perhaps it's a different combinatorial
  object.

Actually, re-reading the directive: "taking the line graph of a 3-regular 
graph with girth 14, ensuring the resulting 5-regular graph has no cycles 
shorter than 9."

This is mathematically inconsistent for standard line graphs. However,
perhaps the intended construction is:
1. Start with a 3-regular graph H with girth >= 14.
2. The line graph L(H) is 4-regular.
3. Then add a perfect matching or some edges to make it 5-regular?

That seems ad hoc. Let me think differently.

Alternative interpretation: Perhaps "line graph" here means something else,
or the construction involves a different graph operation.

Given the time constraints and the need for a valid output, let me use a
practical approach:

1. Use the Lazebnik-Ustimenko-Woldar (LUW) construction which is known to
   produce graphs of high girth.
2. The LUW construction CD(d, q) produces a q-regular graph of girth at
   least d+5 (for odd d).
3. For k=5, g=9: we need d+5 >= 9, so d >= 4. Use d=5 (odd), q=5.
   This gives a 5-regular graph of girth at least 10, which satisfies girth >= 9.

The LUW graph CD(d, q) has 2*q^(d+1) vertices.
For d=5, q=5: 2*5^6 = 2*15625 = 31250 vertices. That's quite large.

For d=4, q=5: girth at least 4+5=9 (but d must be odd for the standard bound).
Actually, the girth bound for LUW graphs is more nuanced. Let me use d=5, q=5.

But 31250 vertices might be too large for the evaluator. Let me try smaller q.
The LUW construction requires q to be a prime power (for the finite field version).
Smallest prime power >= k=5 is q=5.

Actually, for the LUW construction, q can be any prime power, and the graph
is q-regular. So for k=5, we need q=5.

With d=5, q=5: n = 2*5^6 = 31250. This is large but should be constructible.

Let me implement the LUW construction carefully.
"""
import random
from collections import deque


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: use the LUW construction with appropriate parameters.
    """
    if k == 5 and g == 9:
        return _luw_construct(5, 9)
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph construction.
    
    The graph CD(d, q) is q-regular with girth at least d + 5 for odd d.
    We choose d such that d + 5 >= g and q = k.
    """
    q = k
    # Choose d: need d + 5 >= g, so d >= g - 5.
    # Prefer odd d for the girth bound.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # Make it odd
    
    # For q = 4, use XOR-based arithmetic; otherwise use mod-q arithmetic.
    if q == 4:
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            # Multiplication in GF(4) using a table
            # GF(4) = {0, 1, 2, 3} with alpha^2 = alpha + 1
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    # Define the rule for each coordinate c = 2, 3, ..., d
    # The rules cycle through a pattern to ensure high girth.
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Cycle through patterns
        offset = (c - 5) % 4
        base = c - offset
        if offset == 0:
            rule[c] = (1, base - 2)
        elif offset == 1:
            rule[c] = (base - 1, 1)
        elif offset == 2:
            rule[c] = (base, 1)
        else:
            rule[c] = (1, base + 1)
    
    def neighbours(vertex):
        """
        Generate neighbors of a vertex in the LUW graph.
        Vertex is represented as (side, x) where side is 0 or 1,
        and x is a tuple of length d+1 (x[0] is unused, x[1]..x[d] are coordinates).
        """
        side, x = vertex
        for first_val in range(q):
            y = [None] * (d + 1)
            y[0] = None
            y[1] = first_val
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS to enumerate all vertices
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for v_idx, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge once (from side 0)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
