import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)B01
    # M7 = (A01-A11)(B00+B10)

    # Coefficients for A, B, C in each of the 7 products
    # A coeffs (shape 7x4)
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])

    # B coeffs (shape 7x4)
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    # C coeffs (shape 7x4)
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    # Total Rank Calculation:
    # Block i=0: 1x4x8 -> Schoolbook 32 terms
    # Block i=1,2: 2x4x8 -> Split m=4 into two 2s, p=8 into four 2s.
    # Each 2x2x2 block uses 7 terms.
    # Number of 2x2x2 blocks for i=1,2:
    # m blocks: 2 (cols 0-1, 2-3)
    # p blocks: 4 (cols 0-1, 2-3, 4-5, 6-7)
    # Total blocks = 2 * 4 = 8.
    # Terms = 8 * 7 = 56.
    # Total Rank = 32 + 56 = 88.

    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: i=0 (1x4x8 schoolbook)
    for j in range(m):
        for k in range(p):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1

    # Part 2: i=1,2 (2x4x8 Strassen)
    for j_block in range(2):
        j_global_base = j_block * 2
        for k_block in range(4):
            k_global_base = k_block * 2
            for t in range(7):
                # Update U (i=1,2; j=j_block*2 + {0,1})
                for l in range(4):
                    coeff = S_A[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_global_base + loc_j
                        U[g_i * m + g_j, col] += coeff

                # Update V (j=j_block*2 + {0,1}; k=k_block*2 + {0,1})
                for l in range(4):
                    coeff = S_B[t, l]
                    if coeff != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_global_base + loc_j
                        g_k = k_global_base + loc_k
                        V[g_j * p + g_k, col] += coeff

                # Update W (k=k_block*2 + {0,1}; i=1,2)
                for l in range(4):
                    coeff = S_C[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_global_base + loc_k
                        W[g_k * n + g_i, col] += coeff

                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
