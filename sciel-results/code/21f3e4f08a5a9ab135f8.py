import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The tensor is the direct sum of 7 copies of the 2x2 * 2x2 multiplication tensor.
    Rank of 2x2 * 2x2 is 7 (Strassen). Total rank = 7 * 7 = 49 (minimal).
    
    C = A * B where A is 2x2 and B is 2x14.
    B = [B_0 B_1 ... B_6] where each B_t is 2x2.
    C = [A*B_0, A*B_1, ..., A*B_6]
    
    For each block t, compute A*B_t using Strassen's 7-term formula.
    Total rank = 7 * 7 = 49.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks of 2 columns each

    # Strassen terms for 2x2 * 2x2
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1 = (A00+A11)(B00+B11)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2 = (A10+A11)B00
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3 = A00(B01-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4 = A11(B10-B00)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5 = (A00+A01)B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6 = (A10-A00)(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7 = (A01-A11)(B10+B11)
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Quick vectorized verification
    def tensor_check(Uf, Vf, Wf):
        """Check that the decomposition is correct using vectorized ops."""
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback to schoolbook if Strassen construction is incorrect
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

    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
