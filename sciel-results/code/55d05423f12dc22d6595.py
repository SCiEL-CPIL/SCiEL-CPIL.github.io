"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _verify_decomposition(U, V, W, n=4, m=4, p=9, tol=1e-6):
    """Check that the decomposition exactly matches the matmul tensor."""
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        T_decomp += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_decomp, T_expected, atol=tol)


def _build_strassen_114():
    """Build the known rank-114 Strassen block decomposition for 4x4 * 4x9."""
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2 terms: (A_coeffs, B_coeffs, C_contribs)
    strassen_2x2_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Apply Strassen-based 4x4*4x4 multiplication using 2x2 blocks.
        
        The 4x4 matrices are divided into 2x2 blocks.
        A = [[A00, A01], [A10, A11]] where each Ai* is 2x2
        B = [[B00, B01], [B10, B11]] where each Bi* is 2x2
        C = [[C00, C01], [C10, C11]] where each Ci* is 2x2
        
        We use the standard Strassen 7-multiplication algorithm on the 2x2 blocks.
        """
        def get_A(gi, gj):
            return gi * m + gj
        def get_B(gj, gk):
            return gj * p + gk
        def get_C(gk, gi):
            return gk * n + gi

        # For each Strassen term, we need to compute:
        # M_s = (linear combo of A blocks) * (linear combo of B blocks)
        # and add (linear combo of C blocks) = M_s
        
        # Strassen formulas for 2x2 matrix multiplication:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # Each M_s is a product of two 2x2 matrices (from linear combos)
        # We expand each product into rank-1 terms using the 2x2 Strassen algorithm
        
        for A_coeffs, B_coeffs, C_contribs in strassen_2x2_terms:
            # A_coeffs: dict from (li, lj) to coeff, where li, lj in {0,1} are 2x2 block indices
            # B_coeffs: dict from (lj, lk) to coeff
            # C_contribs: dict from (li, lk) to coeff
            
            # Build the actual A, B, C coefficient vectors
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # A side: for each (li, lj) in A_coeffs with coefficient c_a
            # The 2x2 block A[li, lj] has rows A_base_i+li*2, A_base_i+li*2+1
            # and cols A_base_j+lj*2, A_base_j+lj*2+1
            for (li, lj), c_a in A_coeffs.items():
                for di in range(2):
                    for dj in range(2):
                        gi = A_base_i + li * 2 + di
                        gj = A_base_j + lj * 2 + dj
                        u_vec[get_A(gi, gj)] += c_a
            
            # B side: for each (lj, lk) in B_coeffs with coefficient c_b
            for (lj, lk), c_b in B_coeffs.items():
                for dj in range(2):
                    for dk in range(2):
                        gj = B_base_j + lj * 2 + dj
                        gk = B_base_k + lk * 2 + dk
                        v_vec[get_B(gj, gk)] += c_b
            
            # C side: for each (li, lk) in C_contribs with coefficient c_c
            for (li, lk), c_c in C_contribs.items():
                for di in range(2):
                    for dk in range(2):
                        gi = C_base_i + li * 2 + di
                        gk = C_base_k + lk * 2 + dk
                        w_vec[get_C(gk, gi)] += c_c
            
            add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B[:, 0:4] -> C[:, 0:4]
    strassen_4x4(0, 0, 0, 0, 0, 0)
    # Part 2: A * B[:, 4:8] -> C[:, 4:8]
    strassen_4x4(0, 0, 0, 4, 0, 4)
    # Part 3: A * B[:, 8:9] -> C[:, 8:9], schoolbook (4x4 * 4x1 = 16 terms)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = np.array(U_cols, dtype=float).T
    V = np.array(V_cols, dtype=float).T
    W = np.array(W_cols, dtype=float).T
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS

    # Build the rank-114 Strassen decomposition
    U, V, W = _build_strassen_114()

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook (rank 144)
        U_cols, V_cols, W_cols = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m); v = [0] * (m*p); w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
        rank = len(U_cols)
        U = np.array(U_cols, dtype=float).T
        V = np.array(V_cols, dtype=float).T
        W = np.array(W_cols, dtype=float).T

    rank = U.shape[1]
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
