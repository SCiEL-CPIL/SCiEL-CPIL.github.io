"""Program for matrix multiplication 3x4x7 rank minimization.
Uses block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with optimized sub-block decompositions.
    Current best known simple block decomposition rank: 78.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def schoolbook_221(i0, j0, k0):
        """Schoolbook for 2x2 * 2x1 block. Rank 4."""
        for di in range(2):
            for dj in range(2):
                ii = i0 + di
                jj = j0 + dj
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k0], [1.0],
                               [k0*n+ii], [1.0])

    def schoolbook_122(i0, j0, k0):
        """Schoolbook for 1x2 * 2x2 block. Rank 4."""
        for dj in range(2):
            for dk in range(2):
                jj = j0 + dj
                kk = k0 + dk
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def schoolbook_121(i0, j0, k0):
        """Schoolbook for 1x2 * 2x1 block. Rank 2."""
        for dj in range(2):
            jj = j0 + dj
            add_rank1_term([i0*m+jj], [1.0],
                           [jj*p+k0], [1.0],
                           [k0*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        schoolbook_221(i0, j0, k0 + 6)

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook = rank 14.
        """
        for dj in range(2):
            for dk in range(7):
                jj = j0 + dj
                kk = k0 + dk
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25
          M_{1,2,7}: 14
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2:
          M_{3,2,7}: 39
          M_{3,2,7}: 39
        Total: 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the full decomposition
    if time.time() - start_time < deadline:
        m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness
    if time.time() - start_time < deadline - 5.0:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Efficient verification
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                # C[i,k] = sum_j A[i,j] * (sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i])
                # Let M[j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
                # Then C[i,k] = sum_j A[i,j] * M[j,k]
                M_jk = np.zeros(m)
                for r in range(rank):
                    for j in range(m):
                        M_jk[j] += U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
                C_computed[i, k] = np.dot(A[i, :], M_jk)
        
        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-9:
            # Fallback to pure schoolbook
            U_vecs = []
            V_vecs = []
            W_vecs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_rank1_term([i*m+j], [1.0],
                                       [j*p+k], [1.0],
                                       [k*n+i], [1.0])
            rank = len(U_vecs)
            U = np.array(U_vecs).T
            V = np.array(V_vecs).T
            W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
