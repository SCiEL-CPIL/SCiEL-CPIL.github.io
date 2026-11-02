"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
_STRASSEN_2X2 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

_STRASSEN_C_COMBINATION = [
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]

_BLOCK_PRODUCTS = [
    ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
    ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
    ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
    ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
    ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
    ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
]

_C_BLOCK_COMBINATION = [
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _precompute_rank49_decomposition():
    """
    Precompute the rank-49 decomposition using direct combinatorial enumeration.
    Uses base-7 indexing: rank index r = 7*i + s where i is block product, s is Strassen product.
    Returns flat U, V, W arrays of shape (16, 49).
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Precompute lookup tables for efficiency
    # For each block product i and Strassen product s, precompute U/V contributions
    # and for each C-block and C-local entry, precompute W contributions

    for i in range(7):
        a_combo, b_combo = _BLOCK_PRODUCTS[i]
        for s in range(7):
            r = 7 * i + s
            a_local, b_local = _STRASSEN_2X2[s]

            # U contributions
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # V contributions
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            # W contributions
            for (cbr, cbc, c_list) in _C_BLOCK_COMBINATION:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    W[c_col * n + c_row][r] += w_i * v_s

    return U, V, W


# Precompute at module load time
_U49, _V49, _W49 = _precompute_rank49_decomposition()


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Use precomputed rank-49 decomposition
    U = [row[:] for row in _U49]  # shallow copy of rows
    V = [row[:] for row in _V49]
    W = [row[:] for row in _W49]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
