"""Tensor decomposition for matrix multiplication 3x3 * 3x8 using Strassen on 2x2 blocks.

Uses Strassen's rank-7 algorithm on the 2x2 sub-blocks of the 3x3 matrix A,
combined with schoolbook for the remaining 1x3 and 3x1 strips.
Rank = 4*7 + 2*8 + 2*8 + 1*8 = 28 + 16 + 16 + 8 = 68
"""
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' whose matrices
    (shapes (n*m, rank), (m*p, rank), (p*n, rank)) exactly
    decompose the matrix multiplication tensor for
    (n, m, p) = (3, 3, 8) under the W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen's 7 multiplication terms for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U coefficients for A entries (row, col) in 2x2 block:
    #   A11: M1=1, M3=1, M5=1, M6=-1
    #   A12: M5=1, M7=-1
    #   A21: M2=1, M6=1
    #   A22: M1=1, M2=1, M4=1, M7=-1
    #
    # V coefficients for B entries (row, col) in 2x2 block:
    #   B11: M1=1, M2=1, M4=-1, M6=1
    #   B12: M3=1, M5=1, M6=1
    #   B21: M4=1, M6=-1, M7=1
    #   B22: M1=1, M3=-1, M5=1, M7=1
    #
    # W coefficients for C entries (row, col) in 2x2 block:
    #   C11: M1=1, M4=1, M5=-1, M7=1
    #   C12: M3=1, M5=1
    #   C21: M2=1, M4=1
    #   C22: M1=1, M2=-1, M3=1, M6=1

    # Strassen term definitions: list of (U_coeffs, V_coeffs, W_coeffs)
    # Each is a dict mapping (row,col) -> coeff for the 2x2 block
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,0):0, (0,1):0, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):0, (1,0):1, (0,1):0, (1,1):0}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):0, (1,0):0, (0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1, (0,1):0, (1,1):0}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):0, (0,1):1, (1,1):0}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):0, (1,0):0, (0,1):0, (1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):0, (0,1):0, (1,1):1}),
    ]

    # Total rank: 4 Strassen blocks (for 4 pairs of B columns) * 7 + schoolbook
    # Strassen handles A[0:2, 0:2] x B[0:2, 0:8] -> 4 groups of 7 = 28
    # Schoolbook: A[2,0:2] x B[0:2,0:8] = 16, A[0:2,2] x B[2,0:8] = 16, A[2,2] x B[2,0:8] = 8
    rank = 28 + 16 + 16 + 8  # = 68

    U = [[0] * rank for _ in range(n * m)]  # 9 x 68
    V = [[0] * rank for _ in range(m * p)]  # 24 x 68
    W = [[0] * rank for _ in range(p * n)]  # 24 x 68

    col = 0

    # Part 1: Strassen on A[0:2, 0:2] x B[0:2, 0:8]
    # B has 8 columns, so we process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for b_pair in range(4):
        b_start = b_pair * 2  # column offset in B for this pair
        for term_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[term_idx]
            # U: A[0:2, 0:2] entries -> rows 0..3 in U (i*m+j for i in 0,1; j in 0,1)
            for (a_row, a_col), coeff in u_coeffs.items():
                u_idx = a_row * m + a_col  # i*m+j, i in {0,1}, j in {0,1}
                U[u_idx][col] += coeff
            # V: B[0:2, b_start:b_start+2] entries -> rows j*p+k for j in 0,1; k in b_start..b_start+1
            for (b_row, b_col), coeff in v_coeffs.items():
                v_idx = b_row * p + (b_start + b_col)
                V[v_idx][col] += coeff
            # W: C[0:2, 0:2] entries -> rows k*n+i for k in b_start..b_start+1; i in 0,1
            for (c_row, c_col), coeff in w_coeffs.items():
                w_idx = (b_start + c_row) * n + c_col
                W[w_idx][col] += coeff
            col += 1

    # Part 2: Schoolbook for A[2, 0:2] x B[0:2, 0:8]
    # A[2,j] * B[j,k] for j in 0,1; k in 0..7
    for j in range(2):
        for k in range(p):
            U[2 * m + j][col] = 1  # i=2, j
            V[j * p + k][col] = 1  # j, k
            W[k * n + 2][col] = 1  # k, i=2
            col += 1

    # Part 3: Schoolbook for A[0:2, 2] x B[2, 0:8]
    # A[i,2] * B[2,k] for i in 0,1; k in 0..7
    for i in range(2):
        for k in range(p):
            U[i * m + 2][col] = 1  # i, j=2
            V[2 * p + k][col] = 1  # j=2, k
            W[k * n + i][col] = 1  # k, i
            col += 1

    # Part 4: Schoolbook for A[2,2] x B[2, 0:8]
    # A[2,2] * B[2,k] for k in 0..7
    for k in range(p):
        U[2 * m + 2][col] = 1  # i=2, j=2
        V[2 * p + k][col] = 1  # j=2, k
        W[k * n + 2][col] = 1  # k, i=2
        col += 1

    assert col == rank, f"Column count {col} != rank {rank}"

    # Verify the decomposition
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: got {val}")

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    rank = len(result['U'][0])
    print(f"Strassen-based decomposition: rank={rank}, "
          f"U={len(result['U'])}x{len(result['U'][0])}, "
          f"V={len(result['V'])}x{len(result['V'][0])}, "
          f"W={len(result['W'])}x{len(result['W'][0])}, "
          f"time={t1 - t0:.4f}s")
