"""Rank-49 Strassen-based tensor decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
DEADLINE = 140.0
START_TIME = None

# EVOLVE-BLOCK-START
def construct():
    """
    Build the rank-49 exact decomposition for (2,2,14) matrix multiplication.
    
    The 2x14 matrix B is partitioned into 7 consecutive 2x2 column blocks.
    Strassen's rank-7 scheme is applied independently to each block.
    The final factor matrices are assembled with integer entries and a
    lightweight structural sanity check; a schoolbook fallback is provided
    only if the structural check fails.
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    n, m, p = DIMENSIONS
    nm = n * m   # 4
    mp = m * p   # 28
    pn = p * n   # 28

    num_blocks = p // 2  # 7
    rank = 7 * num_blocks  # 49

    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # Strassen terms for 2x2 * 2x2.
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Precompute local index maps for speed
    a_row_maps = []
    for idx, _ in enumerate([0, 1, 2, 3]):
        a_row_maps.append((idx // m, idx % m))
    b_row_maps = []
    for idx, _ in enumerate([0, 1, 2, 3]):
        b_row_maps.append((idx // 2, idx % 2))
    c_row_maps = []
    for idx, _ in enumerate([0, 1, 2, 3]):
        c_row_maps.append((idx // 2, idx % 2))

    # Assemble the three factor matrices
    for t in range(num_blocks):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            for a_idx, coeff in a_coeffs.items():
                a_i, a_j = a_row_maps[a_idx]
                U[a_i * m + a_j, col] = coeff

            for b_idx, coeff in b_coeffs.items():
                b_j, b_col = b_row_maps[b_idx]
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            for c_idx, coeff in c_coeffs.items():
                c_i, c_col = c_row_maps[c_idx]
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Quick structural sanity check: each (i,j,k) slice should have inner product 1
    # We perform a compact check using numpy for a handful of representative cases
    # and trust the algebra if those pass. This avoids O(n*m*p*rank) overhead.
    def quick_check(Uf, Vf, Wf):
        # Check a few (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(0, p, 2):  # check every other k
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not quick_check(U, V, W):
        # Fallback: schoolbook rank 56
        rank = n * m * p  # 56
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Return as lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
