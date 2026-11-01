"""Small 3-regular graph of girth 16 via Lazebnik-Ustimenko-Woldar (LUW) construction.

For k=3 (q=3) and g=16, we use LUW(d,q) with the smallest d such that the
guaranteed girth is at least g. For odd d, girth >= d+5; for even d, girth >= d+4.
With d=11 (odd), girth >= 16. The graph has 2*q^d vertices.
"""

import time

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    q = k

    # Find smallest d such that guaranteed girth >= g
    best_d = None
    for d in range(3, 80):
        if d % 2 == 1:
            min_girth = d + 5
        else:
            min_girth = d + 4
        if min_girth >= g:
            best_d = d
            break

    if best_d is None:
        # Fallback
        if g % 2 == 1:
            best_d = max(3, g - 5)
        else:
            best_d = max(3, g - 4)
        if best_d % 2 != (g % 2):
            best_d += 1

    d = best_d

    # Field operations for GF(q)
    if q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: (x * y) & 1
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: (x * y) % 3
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # LUW rules
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        elif r == 3:
            rule[c] = (1, c0 + 1)

    # Enumerate all point vertices: (0, (x1, x2, ..., xd)) where xi in GF(q)
    # And line vertices: (1, (l1, l2, ..., ld)) where li in GF(q)
    # Total vertices: 2 * q^d

    from itertools import product

    q_power_d = q ** d
    n = 2 * q_power_d

    # Build index mapping
    # Points: 0 to q^d - 1
    # Lines: q^d to 2*q^d - 1

    def point_to_index(coords):
        # coords is tuple of d elements, each in 0..q-1
        idx = 0
        for i in range(d):
            idx = idx * q + coords[i]
        return idx

    def line_to_index(coords):
        idx = 0
        for i in range(d):
            idx = idx * q + coords[i]
        return q_power_d + idx

    def index_to_point(idx):
        coords = []
        for i in range(d):
            coords.append(idx % q)
            idx //= q
        return tuple(coords)

    def index_to_line(idx):
        idx -= q_power_d
        coords = []
        for i in range(d):
            coords.append(idx % q)
            idx //= q
        return tuple(coords)

    # For each point, compute its 3 neighbors (lines)
    # Point p with coords (p1, ..., pd) is connected to line l with coords (l1, ..., ld)
    # iff for all c=2..d: l[c] - p[c] = l[a_c] * p[b_c]
    # where (a_c, b_c) = rule[c]
    # And l1 is free (0, 1, ..., q-1), giving q edges per point.

    edges = []

    # Precompute for speed
    # For a given point p, the line coordinates are determined by l1 and the recurrence
    # l[c] = p[c] + l[a_c] * p[b_c] for c=2..d

    for p_idx in range(q_power_d):
        if time.time() > deadline:
            break
        p_coords = index_to_point(p_idx)
        # p_coords[0] = p1, p_coords[1] = p2, ..., p_coords[d-1] = pd
        # In 1-indexed: p[1]=p_coords[0], p[2]=p_coords[1], ..., p[d]=p_coords[d-1]

        for l1 in range(q):
            # Compute l2..ld
            l = [0] * (d + 1)  # 1-indexed, l[0] unused
            l[1] = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                # l[c] = p[c] + l[a] * p[b]
                # p[c] = p_coords[c-1], p[b] = p_coords[b-1]
                # l[a] = l[a] (already computed since a < c)
                l[c] = add(p_coords[c - 1], mul(l[a], p_coords[b - 1]))
            l_coords = tuple(l[1:d + 1])
            l_idx = line_to_index(l_coords)
            edges.append((p_idx, l_idx))

    return n, edges
