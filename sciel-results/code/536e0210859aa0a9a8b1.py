import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The 2x2 * 2x14 multiplication tensor is a direct sum of 7 independent
    2x2 * 2x2 multiplication tensors. Since each 2x2*2x2 tensor has rank 7
    (Strassen), and rank is additive over direct sums, the total rank is 49.
    This is provably minimal.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks of 2x2

    # Strassen's 7 terms for 2x2 * 2x2:
    # Term: (a_coeffs, b_coeffs, c_coeffs)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B/C indices within block: 00=0, 01=1, 10=2, 11=3
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]

    rank = num_blocks * 7  # 49
    U = [[0] * rank for _ in range(n * m)]   # (4, 49)
    V = [[0] * rank for _ in range(m * p)]   # (28, 49)
    W = [[0] * rank for _ in range(p * n)]   # (28, 49)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j][col] = coeff

            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k][col] = coeff

            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i][col] = coeff

    # Fast verification using vectorized NumPy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i * m + j] * (V_np[j * p + k] * W_np[k * n + i])
                val = prod.sum()
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback to schoolbook rank-56 (should never happen)
        rank = n * m * p  # 56
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
# EVOLVE-BLOCK-END
