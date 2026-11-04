import numpy as np

_N, _M, _P = 3, 5, 5
_RANK = 74

def _make_block(i0, j0, k0, ni, mj, pk):
    """Generate schoolbook tensor factors for a sub-block.
    Returns list of (u_row, v_row, w_row) triples with value 1."""
    entries = []
    for a in range(ni):
        for b in range(mj):
            for c in range(pk):
                entries.append(((i0 + a) * _M + (j0 + b),
                                (j0 + b) * _P + (k0 + c),
                                (k0 + c) * _N + (i0 + a)))
    return entries

_STRASSEN = [
    # Each entry: (U indices with signs, V indices with signs, W indices with signs)
    # M1 = (A00+A11)(B00+B11)
    ([(0,1),(1,1)], [(0,0),(1,1)], [(0,0),(1,1)]),
    # M2 = (A10+A11)B00
    ([(1,0),(1,1)], [(0,0)], [(0,1),(1,1),(-1)]),
    # M3 = A00(B01-B11)
    ([(0,0)], [(0,1),(1,1),(-1)], [(1,0),(1,1)]),
    # M4 = A11(B10-B00)
    ([(1,1)], [(1,0),(0,0),(-1)], [(0,0),(0,1)]),
    # M5 = (A00+A01)B11
    ([(0,0),(0,1)], [(1,1)], [(0,0),(-1),(1,0)]),
    # M6 = (A10-A00)(B00+B01)
    ([(1,0),(0,0),(-1)], [(0,0),(0,1)], [(1,1)]),
    # M7 = (A01-A11)(B10+B11)
    ([(0,1),(1,1),(-1)], [(1,0),(1,1)], [(0,0)]),
]

def construct():
    U = np.zeros((_N * _M, _RANK), dtype=int)
    V = np.zeros((_M * _P, _RANK), dtype=int)
    W = np.zeros((_P * _N, _RANK), dtype=int)

    idx = 0

    # Apply Strassen factors for the top-left 2x2 block
    for u_idx, v_idx, w_idx in _STRASSEN:
        for (row, val) in u_idx:
            U[row, idx] += val
        for (row, val) in v_idx:
            V[row, idx] += val
        for (row, val) in w_idx:
            W[row, idx] += val
        idx += 1

    # Apply schoolbook for remaining blocks
    blocks = [
        (0, 2, 0, 2, 3, 2),   # A12 x B21 -> C11
        (0, 0, 2, 2, 2, 3),   # A11 x B12 -> C12
        (0, 2, 2, 2, 3, 3),   # A12 x B22 -> C12
        (2, 0, 0, 1, 2, 2),   # A21 x B11 -> C21
        (2, 2, 0, 1, 3, 2),   # A22 x B21 -> C21
        (2, 0, 2, 1, 2, 3),   # A21 x B12 -> C22
        (2, 2, 2, 1, 3, 3),   # A22 x B22 -> C22
    ]

    for (i0, j0, k0, ni, mj, pk) in blocks:
        for (u_row, v_row, w_row) in _make_block(i0, j0, k0, ni, mj, pk):
            U[u_row, idx] = 1
            V[v_row, idx] = 1
            W[w_row, idx] = 1
            idx += 1

    if idx != _RANK:
        raise ValueError(f"Rank mismatch: expected {_RANK}, got {idx}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
