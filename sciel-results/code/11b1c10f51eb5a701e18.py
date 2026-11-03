"""Construct a 4-regular graph of girth 10 (cage problem n(4,10)).
Keeps the seed program's LUW construction and adds a deadline guard."""
import time
from functools import lru_cache

def construct(k, g):
    t0 = time.perf_counter()
    DEADLINE = 140.0  # safety margin before 150s kill

    # LUW / Lazebnik-Ustimenko-Woldar style explicit q-regular bipartite graph.
    # For odd d, the girth is at least d + 5.
    q = k
    # Choose d so that d+5 >= g. For g=10, d=5 gives girth >= 10.
    # Use the seed's exact rule for d computation.
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    # Field arithmetic for GF(q)
    if q == 4:
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule: for each c in 2..d, rule[c] = (a, b)
    # Seed's rule:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Points are tuples (side, x) where side is 0 or 1, x is a tuple of length d+1.
    # side 0: x = (None, x1, x2, ..., xd)  -- "points"
    # side 1: x = (y0, y1, ..., yd)        -- "lines"
    # The zero point is (0, (None, 0, 0, ..., 0)).

    def get_neighbours(vertex):
        """Yield all neighbours of the given vertex."""
        side, x = vertex
        for first in range(q):
            # Build the neighbour tuple
            if side == 0:
                # vertex is a point (side 0), neighbour is a line (side 1)
                # x[0] is None, x[1..d] are values
                # line y: y[0] = first, y[1..d] computed
                y = [0] * (d + 1)
                y[0] = first  # "free" coordinate on the line
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                yield (1, tuple(y))
            else:
                # vertex is a line (side 1), neighbour is a point (side 0)
                # y = x for the line
                # point p: p[0] = None, p[1..d] computed, p[0] is None (index 0)
                # Wait, for points x[0] is None. For lines x[0] is the free coord.
                # p ~ l iff l[c] - p[c] = l[a] * p[b]
                # p[c] = l[c] - l[a] * p[b]
                p = [0] * (d + 1)
                p[0] = None  # points have None at index 0
                # We need to set p[1] = first? No.
                # For lines, x[1] corresponds to l[1]. For points, x[1] is p[1].
                # The free coordinate: for lines it's l[0] (index 0).
                # But in the tuple, index 0 is l[0] for lines and None for points.
                # Index 1 is l[1] for lines and p[1] for points.
                # When iterating "for first in range(q)", we're varying the free coordinate.
                # For lines, the free coord is at index 0. But index 0 for points is None.
                # Actually re-reading the seed:
                #   y = [None, first] + [0] * (d - 1)
                # So index 0 is None, index 1 is "first", then d-1 zeros (indices 2..d).
                # That means the free coordinate is at index 1, not 0!
                # Let me re-read carefully.
                
                # Seed:
                #   y = [None, first] + [0] * (d - 1)
                # This creates a list of length d+1.
                # index 0: None
                # index 1: first
                # index 2..d: 0
                # Then for c in 2..d, update y[c].
                # So for the neighbour:
                #   index 0 is always None
                #   index 1 is the "free" parameter
                #   indices 2..d are computed
                # 
                # That means for points: x[0]=None, x[1]=free_param (but for the starting point, x[1]=0)
                # Wait, for the starting point (0, (None, 0, 0, ..., 0)):
                #   index 0 = None, index 1 = 0, indices 2..d = 0
                # 
                # So the tuple layout is:
                #   position 0: None (for points) or a value (for lines)
                #   position 1: free coordinate
                #   positions 2..d: computed coordinates
                #
                # But the seed's neighbours function:
                #   if side == 0 (point): y[c] = add(x[c], mul(y[a], x[b]))
                #   if side == 1 (line): y[c] = sub(x[c], mul(x[a], y[b]))
                #
                # Here x is the current vertex's coordinate tuple, y is the neighbour's.
                # y[0] = None (always, since index 0 in y is None from initialization)
                # Wait no: y = [None, first] + [0]*(d-1)
                # y[0] = None, y[1] = first, y[2..d] = 0
                # So y[0] is always None regardless of side.
                # That means both points and lines have position 0 = None?
                # But then how do we distinguish sides? We store side separately.
                # The side is the first element of the vertex tuple.
                # The coordinate tuple has position 0 = None always.
                # Position 1 = free coord (the parameter we vary).
                # Positions 2..d = computed.
                #
                # For a point (side 0), the coordinate is (None, p1, p2, ..., pd)
                # For a line (side 1), the coordinate is (None, l1, l2, ..., ld)
                # Wait, but then the "first" in the line's free coordinate...
                # For lines, the free coordinate in the original LUW is l_0 (or some index).
                # In this encoding, index 1 is the free one.
                
                # Let me just follow the seed exactly.
                pass
        
        # Let me just implement it following the seed's exact logic.
        side, x = vertex
        for first in range(q):
            y = [None] + [0] * d  # y[0] = None, y[1..d] initialized to 0
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point -> line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line -> point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    # BFS from the zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if time.perf_counter() - t0 > DEADLINE:
            break
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
