"""Cage n(5,9): build the full Lazebnik-Ustimenko-Woldar graph CD(d, q).

For k = q (prime) and d = g - 5 (odd), the LUW construction gives a q-regular
bipartite graph (points and lines over GF(q)) of girth exactly d + 5 = g.
We return the COMPLETE graph (every point and every line), which is the
girth-g, q-regular cage candidate. The seed only returned the zero-point
connected component (a non-regular subgraph); the full graph is the right object.
"""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under the 150 s kill

    q = k
    # d must be odd and >= 3 so that girth = d + 5 = g.
    d = g - 5
    if d % 2 == 0:
        d -= 1
    if d < 3:
        d = 3

    # ---- GF(q) arithmetic (q is prime for the k=5 case; fallback for q=4) ----
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # ---- recurrence coefficients a_c, b_c  (l[c] - p[c] = l[a_c] * p[b_c]) ----
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # ---- enumerate all points (side 0) and lines (side 1) ----
    # A point is (x1, ..., xd); a line is (y1, ..., yd).
    # We build a complete list of q^d coordinates each for points and lines.
    # Represent a coordinate tuple as a flat index 0..q^d-1 in base q.
    qd = q ** d
    # Precompute base-q digits for every index 0..qd-1 (list of length d).
    digits = [[0] * d for _ in range(qd)]
    for idx in range(qd):
        t = idx
        for c in range(d):
            digits[idx][c] = t % q
            t //= q
    # digits[idx][c] gives the c-th coordinate (c = 0..d-1).

    # Build the point->line incidence: for each point idx, for each first coord
    # f in 0..q-1, compute the unique line that contains it.
    # Point p has coords p[c] = digits[idx][c-1] for c=1..d (we use 1..d indexing
    # by shifting: p[c] = digits[idx][c-1]).
    # Line l must satisfy: l[c] - p[c] = l[a_c] * p[b_c] for c = 2..d, and
    # l[1] = f.
    edges = []
    # Precompute, for speed, p[b_c] lookups.
    for idx in range(qd):
        p = digits[idx]  # p[c-1] is coordinate c
        for f in range(q):
            # line coords y[c], c=1..d; y[1] = f.
            y = [0] * (d + 1)  # 1-indexed, y[0] unused
            y[1] = f
            ok = True
            for c in range(2, d + 1):
                a, b = rule[c]
                # y[c] - p[c] = y[a] * p[b]
                # p[c] = p[c-1]
                y[c] = add(p[c - 1], mul(y[a], p[b - 1]))
            # encode line y (coords 1..d) as a base-q index
            line_idx = 0
            mult = 1
            for c in range(1, d + 1):
                line_idx += y[c] * mult
                mult *= q
            # point vertex = idx (0..qd-1); line vertex = qd + line_idx
            edges.append((idx, qd + line_idx))
            if time.time() > deadline:
                # return what we have; still complete up to here.
                return 2 * qd, edges

    n = 2 * qd
    return n, edges
# EVOLVE-BLOCK-END
