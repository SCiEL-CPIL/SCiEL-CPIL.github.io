import time

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        # Determine d based on girth
        # For even girth g=2m, d is often m-1 or similar.
        # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # For g=10: g-5=5 (odd), so g-4=6. d=6.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
        
        n = len(order)
        return n, edges

    # For the specific case of interest, use the known optimal construction
    if k == 4 and g == 10:
        return luw_construct(4, 10)
    else:
        # Fallback for other parameters, though the task focuses on (4,10)
        return luw_construct(k, g)
