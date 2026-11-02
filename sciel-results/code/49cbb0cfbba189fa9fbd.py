"""Improved Lazebnik-Ustimenko-Woldar construction with NumPy vectorization."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = t0 + 130.0

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        table = None

    def add(x, y):
        if q == 4:
            return x ^ y
        return (x + y) % q

    def sub(x, y):
        if q == 4:
            return x ^ y
        return (x - y) % q

    def mul(x, y):
        if q == 4:
            if np.ndim(x) == 0 and np.ndim(y) == 0:
                return table[x, y]
            return table[x, y]
        return (x * y) % q

    # Build rule arrays
    rule_a = np.zeros(d - 1, dtype=np.int64)
    rule_b = np.zeros(d - 1, dtype=np.int64)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(2, d + 1):
        if c == 2:
            rule_a[c-2], rule_b[c-2] = 1, 1
        elif c == 3:
            rule_a[c-2], rule_b[c-2] = 2, 1
        elif c == 4:
            rule_a[c-2], rule_b[c-2] = 1, 2
        else:
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            opts = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule_a[c-2], rule_b[c-2] = opts[r]

    # BFS from zero point
    start = (0, tuple(np.zeros(d + 1, dtype=np.int64)))
    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        x_arr = np.array(x, dtype=np.int64)

        if side == 0:
            # Point -> Lines
            for first in range(q):
                y = np.zeros(d + 1, dtype=np.int64)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c-2]
                    b = rule_b[c-2]
                    y[c] = add(x_arr[c], mul(x_arr[a], y[b]))
                other = (1, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                edges.append((vi, index[other]))
        else:
            # Line -> Points
            y_arr = np.array(x, dtype=np.int64)
            for first in range(q):
                xp = np.zeros(d + 1, dtype=np.int64)
                xp[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c-2]
                    b = rule_b[c-2]
                    xp[c] = sub(y_arr[c], mul(y_arr[a], xp[b]))
                other = (0, tuple(xp))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Edge already counted from point side

    return len(order), edges
# EVOLVE-BLOCK-END
