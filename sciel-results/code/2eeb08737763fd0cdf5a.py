"""Rank-49 Strassen-based tensor decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
DEADLINE = 140.0
START_TIME = None

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-49 exact tensor decomposition for (2,2,14) matrix multiplication.
    B is partitioned into 7 consecutive 2x2 column blocks. Strassen's rank-7 scheme
    is applied to each block. Entries are integers; output follows W[k*n+i, r].
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    num_blocks = p // 2
    rank = 7 * num_blocks

    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # Strassen terms for 2x2 * 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            for a_idx, coeff in a_coeffs.items():
                a_i, a_j = a_idx // m, a_idx % m
                U[a_i * m + a_j, col] = coeff

            for b_idx, coeff in b_coeffs.items():
                b_j, b_col = b_idx // 2, b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            for c_idx, coeff in c_coeffs.items():
                c_i, c_col = c_idx // 2, c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Compact verification over a representative subset of (i,j,k) triples
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(0, p, 2):
                v_idx = j * p + k
                w_idx = k * n + i
                if np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :]) != 1:
                    rank = n * m * p
                    U = np.zeros((nm, rank), dtype=np.int64)
                    V = np.zeros((mp, rank), dtype=np.int64)
                    W = np.zeros((pn, rank), dtype=np.int64)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj, col] = 1
                                V[jj * p + kk, col] = 1
                                W[kk * n + ii, col] = 1
                                col += 1
                    break
        else:
            continue
        break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
