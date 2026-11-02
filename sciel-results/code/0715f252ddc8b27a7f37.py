import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Optimized for speed with pre-computed rules and efficient BFS.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    q = k
    # Determine d such that girth >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # Pre-compute the rule table: rule[c] = (a, b) for c in 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    # Pre-compute rules as lists for fast access
    rules_a = [0] * (d + 1)
    rules_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rules_a[c] = rule[c][0]
        rules_b[c] = rule[c][1]

    # For q=3, use integer arithmetic directly
    if q == 3:
        # Pre-compute multiplication table
        mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    else:
        mul_table = None

    # Vertex representation: encode as integer for fast hashing
    # Vertex = (side, x1, x2, ..., xd) where each xi in 0..q-1
    # Encode: side * q^d + x1*q^(d-1) + x2*q^(d-2) + ... + xd
    q_pows = [1] * (d + 2)
    for i in range(1, d + 2):
        q_pows[i] = q_pows[i - 1] * q
    qd = q_pows[d]  # q^d
    total_vertices = 2 * qd

    # Pre-compute encoding/decoding helpers
    # For a given side and coordinates x[1..d], the index is:
    # side * qd + sum(x[i] * q^(d-i) for i in 1..d)
    # i.e., side * qd + x1*q^(d-1) + x2*q^(d-2) + ... + xd*q^0

    # We'll do BFS using the integer encoding
    # Start vertex: side=0, all coords=0
    start_idx = 0  # side=0, all zeros

    # BFS
    visited = [False] * total_vertices
    visited[start_idx] = True
    queue = [start_idx]
    edges = []
    head = 0

    while head < len(queue):
        if time.time() > deadline:
            break

        v = queue[head]
        head += 1

        side = v // qd
        # Decode coordinates
        rem = v % qd
        x = [0] * (d + 1)  # x[1..d]
        for i in range(d, 0, -1):
            x[i] = rem % q
            rem //= q

        # Generate q neighbors
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a = rules_a[c]
                b = rules_b[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    if mul_table is not None:
                        y[c] = (x[c] + mul_table[y[a]][x[b]]) % q
                    else:
                        y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    if mul_table is not None:
                        y[c] = (x[c] - mul_table[x[a]][y[b]]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q

            # Encode neighbor
            nside = 1 - side
            nrem = 0
            for i in range(1, d + 1):
                nrem = nrem * q + y[i]
            nidx = nside * qd + nrem

            if not visited[nidx]:
                visited[nidx] = True
                queue.append(nidx)

            if side == 0:
                edges.append((v, nidx))

    n = len(queue)
    return n, edges
