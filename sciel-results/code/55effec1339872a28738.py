import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the rank-11 decomposition terms.
    # Each term is a tuple: (u_indices, v_indices, w_indices)
    # u_indices: list of (i, j, coeff) for U[i*m+j]
    # v_indices: list of (j, k, coeff) for V[j*p+k]
    # w_indices: list of (k, i, coeff) for W[k*n+i]
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # Term 2: (A10+A11)B00 -> C10, -C11
    # Term 3: A00(B01-B11) -> C01, C11
    # Term 4: A11(B10-B00) -> C00, C10
    # Term 5: (A00+A01)B11 -> -C00, C01
    # Term 6: (A10-A00)(B00+B01) -> C11
    # Term 7: (A01-A11)(B10+B11) -> C00
    # Term 8: A00B02 -> C02
    # Term 9: A01B12 -> C02
    # Term 10: A10B02 -> C12
    # Term 11: A11B12 -> C12

    terms = [
        # (u_entries, v_entries, w_entries)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 0, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ([(0, 0, 1)], [(0, 2, 1)], [(2, 0, 1)]),
        ([(0, 1, 1)], [(1, 2, 1)], [(2, 0, 1)]),
        ([(1, 0, 1)], [(0, 2, 1)], [(2, 1, 1)]),
        ([(1, 1, 1)], [(1, 2, 1)], [(2, 1, 1)]),
    ]

    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for i, j, val in u_entries:
            U[i * m + j][r] += val
        for j, k, val in v_entries:
            V[j * p + k][r] += val
        for k, i, val in w_entries:
            W[k * n + i][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
