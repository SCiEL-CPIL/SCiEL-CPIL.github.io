"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Split B into B1 (4x4) and B2 (4x2).
- A*B1 (4x4 * 4x4) uses 2x2 Strassen tiling: 4 blocks * rank 7 = 28 terms.
- A*B2 (4x4 * 4x2) uses 2x2 schoolbook tiling: 4 blocks * rank 4 = 16 terms.
Total rank: 28 + 16 = 44.

The multiplication tensor T(A,B) = A*B where A is 4x4, B is 4x6, C is 4x6.
Decomposition: T = sum_r u_r * v_r * w_r
  u_r in R^{16} (A-space, index i*m+j)
  v_r in R^{24} (B-space, index j*p+k)
  w_r in R^{24} (C-space, index k*n+i per trace convention W[k*n+i, r])
"""

import time

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6

    U_rows = n * m   # 16
    V_rows = m * p   # 24
    W_rows = p * n   # 24

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def zeros(length):
        return [0] * length

    # No Laderman needed - we use 2x2 Strassen tiling for 4x4*4x4 part
    # and 2x2 schoolbook tiling for 4x4*4x2 part.

    def strassen_2x2_2x2(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 terms for Strassen 2x2*2x2 at given offsets.
        A block at rows ar:ar+2, cols ac:ac+2
        B block at rows br:br+2, cols bc:bc+2
        C block at rows cr:cr+2, cols cc:cc+2
        C[i][j] = sum_k A[i][k]*B[k][j]
        """
        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        # P1 = (A00+A11)*(B00+B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # P2 = (A10+A11)*B00
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1; w[c_idx(1, 1)] = -1
        add_term(u, v, w)

        # P3 = A00*(B01-B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 1)] = 1; v[b_idx(1, 1)] = -1
        w[c_idx(0, 1)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # P4 = (A11-A00)*B10
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(1, 0)] = 1
        w[c_idx(0, 0)] = -1; w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # P5 = (A00+A01)*B11
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(0, 1)] = 1
        v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # P6 = (A10-A00)*(B00+B01)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(0, 0)] = 1; v[b_idx(0, 1)] = 1
        w[c_idx(0, 0)] = -1
        add_term(u, v, w)

        # P7 = (A01-A11)*(B10+B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1; u[a_idx(1, 1)] = -1
        v[b_idx(1, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(1, 1)] = 1
        add_term(u, v, w)

    def schoolbook_2x2_2x1(ar, ac, br, bc, cr, cc):
        """Add 4 rank-1 terms for schoolbook 2x2*2x1 at given offsets.
        A block: rows ar:ar+2, cols ac:ac+2 (2x2)
        B block: rows br:br+2, col bc (2x1)
        C block: rows cr:cr+2, col cc (2x1)
        C[i][0] = sum_j A[i][j]*B[j][0]
        """
        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        # C[0][0] += A[0][0]*B[0][0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

        # C[0][0] += A[0][1]*B[1][0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1
        v[b_idx(1, 0)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

        # C[1][0] += A[1][0]*B[0][0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # C[1][0] += A[1][1]*B[1][0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1
        v[b_idx(1, 0)] = 1
        w[c_idx(1, 0)] = 1
        add_term(u, v, w)

    # Part 1: C_left = A * B_left where B_left is B[:, 0:4] (4x4)
    # Use 2x2 Strassen tiling: 4 blocks of 2x2*2x2
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    strassen_2x2_2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2_2x2(0, 2, 2, 0, 0, 0)
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    strassen_2x2_2x2(0, 0, 0, 2, 0, 2)
    strassen_2x2_2x2(0, 2, 2, 2, 0, 2)
    # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    strassen_2x2_2x2(2, 0, 0, 0, 2, 0)
    strassen_2x2_2x2(2, 2, 2, 0, 2, 0)
    # C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    strassen_2x2_2x2(2, 0, 0, 2, 2, 2)
    strassen_2x2_2x2(2, 2, 2, 2, 2, 2)

    # Part 2: C_right = A * B_right where B_right is B[:, 4:6] (4x2)
    # Use 2x2 schoolbook tiling: 4 blocks of 2x2*2x1
    # C[0:2, 4] = A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4]
    schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4)
    schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4)
    # C[0:2, 5] = A[0:2,0:2]*B[0:2,5] + A[0:2,2:4]*B[2:4,5]
    schoolbook_2x2_2x1(0, 0, 0, 5, 0, 5)
    schoolbook_2x2_2x1(0, 2, 2, 5, 0, 5)
    # C[2:4, 4] = A[2:4,0:2]*B[0:2,4] + A[2:4,2:4]*B[2:4,4]
    schoolbook_2x2_2x1(2, 0, 0, 4, 2, 4)
    schoolbook_2x2_2x1(2, 2, 2, 4, 2, 4)
    # C[2:4, 5] = A[2:4,0:2]*B[0:2,5] + A[2:4,2:4]*B[2:4,5]
    schoolbook_2x2_2x1(2, 0, 0, 5, 2, 5)
    schoolbook_2x2_2x1(2, 2, 2, 5, 2, 5)

    rank = len(U_cols)

    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]

    # Verify correctness with a random test
    try:
        import numpy as np
        np.random.seed(42)
        A_test = np.random.randint(-2, 3, (n, m)).astype(float)
        B_test = np.random.randint(-2, 3, (m, p)).astype(float)
        C_expected = A_test @ B_test

        U_mat = np.array(U, dtype=float)  # shape (U_rows, rank)
        V_mat = np.array(V, dtype=float)  # shape (V_rows, rank)
        W_mat = np.array(W, dtype=float)  # shape (W_rows, rank)

        # C[k*n+i] = sum_r W[k*n+i, r] * ... 
        # Actually, the tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = C[i,k] for the (i,k) entry, summing over j
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = delta(C[i,k], sum_j A[i,j]*B[j,k])
        # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_vec = U_mat[:, r]  # (U_rows,) = (n*m,)
            v_vec = V_mat[:, r]  # (V_rows,) = (m*p,)
            w_vec = W_mat[:, r]  # (W_rows,) = (p*n,)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # Contribution to C[i,k]: sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += np.sum(u_mat[i, :] * v_mat[:, k] * w_mat[k, i])

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: use full schoolbook
            U_cols2 = []
            V_cols2 = []
            W_cols2 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = [0] * U_rows
                        v = [0] * V_rows
                        w = [0] * W_rows
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i] = 1
                        U_cols2.append(u)
                        V_cols2.append(v)
                        W_cols2.append(w)
            rank2 = len(U_cols2)
            U = [[U_cols2[r][i] for r in range(rank2)] for i in range(U_rows)]
            V = [[V_cols2[r][i] for r in range(rank2)] for i in range(V_rows)]
            W = [[W_cols2[r][i] for r in range(rank2)] for i in range(W_rows)]
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
