"""Cayley-graph construction over PSL(2,11) for the (5,9) cage problem."""
# EVOLVE-BLOCK-START

def _matmul(A, B):
    r, c, m = A[0][0], A[0][0], B[0][0]
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)],
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)],
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)],
    ]

def _matinv(A):
    a, b, c = A[0][0], A[0][1], A[0][2]
    d, e, f = A[1][0], A[1][1], A[1][2]
    g, h, i = A[2][0], A[2][1], A[2][2]
    det = (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
    det_inv = pow(det, -1, 11)
    return [
        [((e * i - f * h) * det_inv) % 11, (-(b * i - c * h)) * det_inv % 11, (b * f - c * e) * det_inv % 11],
        [(-(d * i - f * g)) * det_inv % 11, (a * i - c * g) * det_inv % 11, (-(a * f - c * d)) * det_inv % 11],
        [(d * h - e * g) * det_inv % 11, (-(a * h - b * g)) * det_inv % 11, (a * e - b * d) * det_inv % 11],
    ]

def _matkey(A):
    return (A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2], A[2][0], A[2][1], A[2][2])

def construct(k, g):
    # Generators of the (11,12) cage connection set in PSL(2,11).
    a = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    b = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    c = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    d = [[0, 1, 1], [0, 0, 1], [1, 0, 0]]
    e = [[1, 0, 0], [1, 0, 1], [0, 1, 0]]

    # Enumerate the group by BFS from the identity using the (undirected) connection set.
    S = [a, b, c, d, e]
    Sinv = [_matinv(x) for x in S]
    moves = S + Sinv
    start = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    index = {_matkey(start): 0}
    order = [start]
    for v in order:
        for m in moves:
            nv = _matmul(v, m)
            key = _matkey(nv)
            if key not in index:
                index[key] = len(order)
                order.append(nv)

    n = len(order)
    # Build the simple undirected edge set from the 5 generators (each is an involution).
    edge_set = set()
    for u in range(n):
        for gmat in S:
            key = _matkey(_matmul(order[u], gmat))
            v = index[key]
            if u < v:
                edge_set.add((u, v))
    edges = sorted(edge_set)
    return n, edges
# EVOLVE-BLOCK-END
