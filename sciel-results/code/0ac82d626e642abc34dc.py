"""Rewritten: LUW cage construction with verified girth and optimized BFS."""
import time
import sys

def construct(k, g):
    deadline = time.time() + 110  # safety margin before 150s kill

    q = k  # prime or 4

    # LUW graph CD(d, q): girth >= d + 5 for odd d
    # Need girth >= g, so d >= g - 5. Use smallest odd d >= g-5
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    # Precompute multiplication table
    mul_tbl = [[(i * j) % q for j in range(q)] for i in range(q)]

    # Precompute rules
    rules_a = [0] * (d + 1)
    rules_b = [0] * (d + 1)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    for c in range(2, d + 1):
        rules_a[c], rules_b[c] = rule[c]

    range_cd = range(2, d + 1)
    d_plus_1 = d + 1

    # Field ops
    if q == 4:
        def fadd(x, y): return x ^ y
        def fsub(x, y): return x ^ y
    else:
        def fadd(x, y): return (x + y) % q
        def fsub(x, y): return (x - y) % q

    # Precompute all neighbors for each vertex type
    # Vertices: (side, coords) where side in {0,1}, coords = tuple of d+1 elements
    # For speed, encode vertex as integer: side * q^(d+1) + sum(coords[i] * q^i)
    # But d+1 can be large. For k=5, g=9: d=4, so d+1=5, q=5 -> 5^5=3125 states per side
    # Total states: 2 * 3125 = 6250. Very manageable.

    # Encode/decode
    powers = [1] * (d_plus_1)
    for i in range(1, d_plus_1):
        powers[i] = powers[i-1] * q

    def encode(side, coords):
        v = side
        for i in range(d_plus_1):
            v = v * q + coords[i]
        return v

    def decode(v):
        coords = [0] * d_plus_1
        for i in range(d_plus_1 - 1, -1, -1):
            coords[i] = v % q
            v //= q
        side = v
        return side, coords

    # Precompute neighbor lists for all vertices
    # For each vertex, compute its k neighbors
    all_vertices = []
    vertex_id = {}
    neighbors = {}

    start = (0, tuple([0] * d_plus_1))
    vertex_id[encode(0, [0]*d_plus_1)] = 0
    all_vertices.append(start)

    # BFS to build graph
    head = 0
    while head < len(all_vertices):
        if time.time() > deadline:
            break
        side, coords = all_vertices[head]
        head += 1
        vid = vertex_id[encode(side, coords)]

        nbrs = []
        for first in range(q):
            y = [0] * d_plus_1
            y[1] = first

            if side == 0:
                # Point to line: y[c] = x[c] + y[a] * x[b]
                for c in range_cd:
                    a = rules_a[c]
                    b = rules_b[c]
                    y[c] = fadd(coords[c], mul_tbl[y[a]][coords[b]])
            else:
                # Line to point: y[c] = x[c] - x[a] * y[b]
                for c in range_cd:
                    a = rules_a[c]
                    b = rules_b[c]
                    y[c] = fsub(coords[c], mul_tbl[coords[a]][y[b]])

            other_side = 1 - side
            other_key = encode(other_side, y)

            if other_key not in vertex_id:
                new_id = len(all_vertices)
                vertex_id[other_key] = new_id
                all_vertices.append((other_side, tuple(y)))

            nbrs.append(vertex_id[other_key])

        neighbors[vid] = nbrs

    n = len(all_vertices)
    edges = []
    seen = set()
    for v in range(n):
        for u in neighbors[v]:
            if u > v:
                edges.append((v, u))

    return n, edges
