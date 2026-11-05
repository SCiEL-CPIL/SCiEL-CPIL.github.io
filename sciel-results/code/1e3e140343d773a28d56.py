# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # --- LUW Construction (Lazebnik-Ustimenko-Woldar) ---
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            if g <= 5:
                d = 3
            else:
                d = 3
                while True:
                    girth_est = d + 5 if d % 2 == 1 else d + 4
                    if girth_est >= g:
                        break
                    d += 1

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        # Precompute multiplication and addition tables for speed
        mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
        add_table = [[add(i, j) for j in range(q)] for i in range(q)]
        sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

        # For point-side vertices (side=0), neighbor is line-side (side=1)
        # y[1] = first (free variable), y[c] = x[c] + y[a]*x[b] for c=2..d
        # For line-side vertices (side=1), neighbor is point-side (side=0)
        # y[1] = first (free variable), y[c] = x[c] - x[a]*y[b] for c=2..d

        def compute_neighbors_point(x):
            """x is a point (side 0). Returns list of line neighbors (side 1)."""
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            """x is a line (side 1). Returns list of point neighbors (side 0)."""
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        
        for vertex in order:
            if time.time() > deadline - 20:
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        
        # Check if the graph is complete (all expected vertices reached)
        # For LUW CD(d,q), expected size is 2*q^(d) if fully connected
        # But we just return what we have
        return len(order), edges

    # Try multiple d values and pick the smallest valid graph
    best_n = None
    best_edges = None

    # Compute the range of d to try
    if g <= 5:
        d_min = 3
    else:
        d_min = 3
        while True:
            girth_est = d_min + 5 if d_min % 2 == 1 else d_min + 4
            if girth_est >= g:
                break
            d_min += 1

    # Try d_min, d_min+1, d_min+2 (if time permits)
    for d_try in range(d_min, d_min + 3):
        if time.time() > deadline - 30:
            break
        try:
            n_try, edges_try = luw_construct(k, g, d_try)
            if n_try < 50000:  # sanity check
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges_try
        except Exception:
            pass

    if best_n is not None:
        return best_n, best_edges

    # Fallback: basic LUW
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
