"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook(n, m, p):
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
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition.
        U indexed by (i,j) with i in {0,1}, j in {0,1} -> index i*2+j.
        V indexed by (j,k) with j in {0,1}, k in {0,1} -> index j*2+k.
        W indexed by (k,i) with k in {0,1}, i in {0,1} -> index k*2+i.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22

        # M2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21
        W[3, 1] = -1  # C22

        # M3: A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22

        # M4: A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21

        # M5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # C11
        W[1, 4] = 1  # C12

        # M6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22

        # M7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11

        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j*p+k,r] == 1."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i * m + j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k * n + i, r] * M_r[i] * N_r[k]

        return np.allclose(C_decomp, C_expected, atol=1e-10)

    def strassen_block(n_full, m_sub, p_sub, m_start, p_start):
        """Strassen (2x2x2) embedded in full n_full, m, p.
        m_sub=2, p_sub=2. Returns (U_block, V_block, W_block) of rank 7.
        """
        Us, Vs, Ws = strassen_2x2x2()
        rank_s = 7
        U_block = np.zeros((n_full * m_sub, rank_s))
        V_block = np.zeros((m_sub * p_sub, rank_s))
        W_block = np.zeros((p_sub * n_full, rank_s))

        for r in range(rank_s):
            for i in range(n_full):
                for j in range(m_sub):
                    U_block[i * m_sub + (m_start + j), r] = Us[i * m_sub + j, r] if i < 2 else 0
            for j in range(m_sub):
                for k in range(p_sub):
                    V_block[(m_start + j) * p_sub + (p_start + k), r] = Vs[j * p_sub + k, r]
            for k in range(p_sub):
                for i in range(n_full):
                    W_block[(p_start + k) * n_full + i, r] = Ws[k * n_full + i, r]

        return U_block, V_block, W_block

    def schoolbook_block(n_full, m_sub, p_sub, m_start, p_start):
        """Schoolbook n_sub x m_sub x p_sub block embedded in full n_full, m, p.
        Returns (U_block, V_block, W_block).
        """
        rank_s = n_full * m_sub * p_sub
        U_block = np.zeros((n_full * m_sub, rank_s))
        V_block = np.zeros((m_sub * p_sub, rank_s))
        W_block = np.zeros((p_sub * n_full, rank_s))

        col = 0
        for i in range(n_full):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_block[i * m_sub + (m_start + j), col] = 1
                    V_block[(m_start + j) * p_sub + (p_start + k), col] = 1
                    W_block[(p_start + k) * n_full + i, col] = 1
                    col += 1

        return U_block, V_block, W_block

    def compute_block_decomposition():
        """Block-diagonal decomposition of 3x5x6.
        Split m=5 into [2,3], p=6 into [2,4].
        Blocks:
          (0,0): 3x2x2 -> Strassen rank 7
          (0,1): 3x2x4 -> two Strassen (2x2x2) blocks, rank 14
          (1,0): 3x3x2 -> schoolbook rank 18
          (1,1): 3x3x4 -> schoolbook rank 36
        Total: 7 + 14 + 18 + 36 = 75
        """
        m_splits = [2, 3]
        p_splits = [2, 4]
        n_sub = 3

        U_parts = []
        V_parts = []
        W_parts = []

        for j_idx, m_block in enumerate(m_splits):
            m_start = sum(m_splits[:j_idx])
            for k_idx, p_block in enumerate(p_splits):
                p_start = sum(p_splits[:k_idx])

                if m_block == 2 and p_block == 2:
                    # 3x2x2: Strassen
                    U_b, V_b, W_b = strassen_block(n_sub, m_block, p_block, m_start, p_start)
                elif m_block == 2 and p_block == 4:
                    # 3x2x4: two Strassen (2x2x2) blocks
                    U_b = np.zeros((n_sub * m_block, 14))
                    V_b = np.zeros((m_block * p_block, 14))
                    W_b = np.zeros((p_block * n_sub, 14))
                    Us, Vs, Ws = strassen_2x2x2()
                    for i in range(n_sub):
                        for r in range(7):
                            # First Strassen block (k=0,1)
                            U_b[i * m_block + 0, i * 7 + r] = Us[i * 2 + 0, r] if i < 2 else 0
                            U_b[i * m_block + 1, i * 7 + r] = Us[i * 2 + 1, r] if i < 2 else 0
                            V_b[0 * p_block + 0, i * 7 + r] = Vs[0 * 2 + 0, r]
                            V_b[0 * p_block + 1, i * 7 + r] = Vs[0 * 2 + 1, r]
                            V_b[1 * p_block + 0, i * 7 + r] = Vs[1 * 2 + 0, r]
                            V_b[1 * p_block + 1, i * 7 + r] = Vs[1 * 2 + 1, r]
                            W_b[0 * n_sub + i, i * 7 + r] = Ws[0 * 2 + 0, r]
                            W_b[1 * n_sub + i, i * 7 + r] = Ws[0 * 2 + 1, r]
                            W_b[2 * n_sub + i, i * 7 + r] = Ws[1 * 2 + 0, r]
                            W_b[3 * n_sub + i, i * 7 + r] = Ws[1 * 2 + 1, r]
                    # Second Strassen block (k=2,3)
                    for i in range(n_sub):
                        for r in range(7):
                            U_b[i * m_block + 0, i * 7 + 7 + r] = Us[i * 2 + 0, r] if i < 2 else 0
                            U_b[i * m_block + 1, i * 7 + 7 + r] = Us[i * 2 + 1, r] if i < 2 else 0
                            V_b[0 * p_block + 2, i * 7 + 7 + r] = Vs[0 * 2 + 0, r]
                            V_b[0 * p_block + 3, i * 7 + 7 + r] = Vs[0 * 2 + 1, r]
                            V_b[1 * p_block + 2, i * 7 + 7 + r] = Vs[1 * 2 + 0, r]
                            V_b[1 * p_block + 3, i * 7 + 7 + r] = Vs[1 * 2 + 1, r]
                            W_b[2 * n_sub + i, i * 7 + 7 + r] = Ws[0 * 2 + 0, r]
                            W_b[3 * n_sub + i, i * 7 + 7 + r] = Ws[0 * 2 + 1, r]
                            W_b[4 * n_sub + i, i * 7 + 7 + r] = Ws[1 * 2 + 0, r]
                            W_b[5 * n_sub + i, i * 7 + 7 + r] = Ws[1 * 2 + 1, r]
                else:
                    # Schoolbook
                    U_b, V_b, W_b = schoolbook_block(n_sub, m_block, p_block, m_start, p_start)

                U_parts.append(U_b)
                V_parts.append(V_b)
                W_parts.append(W_b)

        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)

        return U, V, W

    # Try the block decomposition
    U, V, W = compute_block_decomposition()
    rank_total = U.shape[1]

    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
