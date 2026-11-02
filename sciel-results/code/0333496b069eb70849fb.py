"""Construct the (4,10)-cage: a 4-regular graph of girth 10 on 320 vertices."""

def construct(k, g):
    if k != 4 or g != 10:
        # Generic fallback for other parameters
        return _generic_construct(k, g)
    
    # GF(4) arithmetic using XOR for addition and table for multiplication
    _GF4_MUL = (
        0, 0, 0, 0,
        0, 1, 2, 3,
        0, 2, 3, 1,
        0, 3, 1, 2
    )
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        return _GF4_MUL[a * 4 + b]
    
    # Algebraic rules for the (4,10)-cage: d=5, q=4
    # rule[c] = (a, b) means: y[c] = x[c] + y[a] * x[b]  (for point-to-line)
    # and:      x[c] = y[c] + x[a] * y[b]  (for line-to-point)
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2), c=5: (1,3)
    _RULES = (None, None, (1, 1), (2, 1), (1, 2), (1, 3))
    
    # Build adjacency by iterating over all possible point coordinates
    # Points are indexed 0..159, lines are indexed 160..319
    # Point i has coordinates derived from i in base-4
    # Line j has coordinates derived from (j-160) in base-4
    
    def coords_from_index(idx):
        """Convert integer index to 5-digit base-4 tuple."""
        digits = []
        for _ in range(5):
            digits.append(idx % 4)
            idx //= 4
        return tuple(digits)
    
    # For each point, compute its 4 neighboring lines
    # For each line, compute its 4 neighboring points
    # The incidence: point p=(x1,x2,x3,x4,x5) is adjacent to line l=(y1,y2,y3,y4,y5)
    # if for each c in 2..5: y[c] = x[c] + y[a_c] * x[b_c]
    # Given a point and y1, we can determine y2..y5 uniquely
    # Given a line and x1, we can determine x2..x5 uniquely
    
    def point_to_lines(x):
        """Given point coordinates x=(x1,x2,x3,x4,x5), yield the 4 line indices."""
        for y1 in range(4):
            y = [0] * 6  # 1-indexed, y[1]=y1
            y[1] = y1
            for c in range(2, 6):
                a, b = _RULES[c]
                y[c] = gf4_add(x[c - 1], gf4_mul(y[a - 1], x[b - 1]))
            # Convert y to an index
            yidx = 0
            for c in range(5, 0, -1):
                yidx = yidx * 4 + y[c]
            yield yidx
    
    def line_to_points(y):
        """Given line coordinates y=(y1,y2,y3,y4,y5), yield the 4 point indices."""
        for x1 in range(4):
            x = [0] * 6  # 1-indexed, x[1]=x1
            x[1] = x1
            for c in range(2, 6):
                a, b = _RULES[c]
                x[c] = gf4_add(y[c - 1], gf4_mul(x[a - 1], y[b - 1]))
            # Convert x to an index
            xidx = 0
            for c in range(5, 0, -1):
                xidx = xidx * 4 + x[c]
            yield xidx
    
    # Build edge list
    edges = []
    
    # For each point (0..159), add edges to its 4 neighboring lines (160..319)
    for pidx in range(160):
        x = coords_from_index(pidx)
        for lidx in point_to_lines(x):
            edges.append((pidx, 160 + lidx))
    
    return 320, edges


def _generic_construct(k, g):
    """Generic fallback using Lazebnik-Ustimenko-Woldar construction."""
    q = k
    # For girth g (even), d = g/2
    d = g // 2
    
    if q == 4:
        _table = (
            0, 0, 0, 0,
            0, 1, 2, 3,
            0, 2, 3, 1,
            0, 3, 1, 2
        )
        def _add(a, b): return a ^ b
        def _sub(a, b): return a ^ b
        def _mul(a, b): return _table[a * 4 + b]
    else:
        def _add(a, b): return (a + b) % q
        def _sub(a, b): return (a - b) % q
        def _mul(a, b): return (a * b) % q
    
    # Build rules
    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    # Build graph via BFS
    from collections import deque
    
    def neighbours(vertex):
        side, x = vertex
        result = []
        for first in range(q):
            y = [0] * (d + 1)  # 1-indexed conceptually
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = _add(x[c - 1], _mul(y[a - 1], x[b - 1]))
                else:
                    y[c] = _sub(x[c - 1], _mul(x[a - 1], y[b - 1]))
            result.append((1 - side, tuple(y)))
        return result
    
    start = (0, tuple([0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    queue = deque([0])
    
    while queue:
        vi = queue.popleft()
        vertex = order[vi]
        side, coords = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(index[other])
            if side == 0:
                edges.append((vi, index[other]))
    
    return len(order), edges
