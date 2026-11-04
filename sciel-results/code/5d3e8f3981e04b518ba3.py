"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: Use structured decompositions for sub-tensors.
    # For i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}:
    #   - Strassen on 2x2x2 blocks for k in {0,1}, {2,3}: 4 blocks * 7 = 28
    #   - Rank 2 blocks for k=4, j in {0,1} and j in {2,3}: 2 * 2 = 4
    #   Total for i in {0,1}: 32
    # For i=2, j in {0,1,2,3}, k in {0,1,2,3,4}:
    #   - Rank 2 blocks for j in {0,1}, k in {0,1}, {2,3}: 2 * 2 = 4
    #   - Rank 2 blocks for j in {2,3}, k in {0,1}, {2,3}: 2 * 2 = 4
    #   - Rank 1 blocks for k=4, j in {0,1} and j in {2,3}: 2 * 1 = 2
    #   Total for i=2: 10
    # Grand total: 32 + 10 = 42

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a Strassen block for sub-problem at (i_offset, j_offset, k_offset)
    def add_strassen_block(i_off, j_off, k_off):
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off+ii)*m + (j_off+jj)] = U_s[ii*2+jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off+jj)*p + (k_off+kk)] = V_s[jj*2+kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off+kk)*n + (i_off+ii)] = W_s[kk*2+ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Helper to add rank-2 block for 1x2x2 at (i, j0, j1, k0, k1)
    def add_rank2_block_1x2x2(i, j0, j1, k0, k1):
        # Two terms
        for c in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            if c == 0:
                U_col[i*m + j0] = 1.0
                V_col[j0*p + k0] = 1.0
                V_col[j0*p + k1] = 1.0
                W_col[k0*n + i] = 1.0
                W_col[k1*n + i] = 1.0
            else:
                U_col[i*m + j1] = 1.0
                V_col[j1*p + k0] = 1.0
                V_col[j1*p + k1] = 1.0
                W_col[k0*n + i] = 1.0
                W_col[k1*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Helper to add rank-1 block for 1x2x1 at (i, j0, j1, k)
    def add_rank1_block_1x2x1(i, j0, j1, k):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[i*m + j0] = 1.0
        U_col[i*m + j1] = 1.0
        V_col[j0*p + k] = 1.0
        V_col[j1*p + k] = 1.0
        W_col[k*n + i] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}:
    #   Strassen on 2x2x2 blocks for j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    #   Strassen on 2x2x2 blocks for j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    #   Strassen on 2x2x2 blocks for j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    #   Strassen on 2x2x2 blocks for j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    #   Rank 2 blocks for k=4, j in {0,1}
    add_rank2_block_1x2x2(0, 0, 1, 4, 4)  # This is 1x2x1, not 1x2x2. Let me fix.
    # Actually, for k=4, it's a 1x2x1 tensor (only one k value). So rank 1.
    # Let me reconsider. For k=4, j in {0,1}, it's a 1x2x1 tensor. Rank 1.
    # So I should use rank-1 blocks.

    # Let me restart the construction with the correct ranks.
    U_cols = []
    V_cols = []
    W_cols = []

    # For i in {0,1}:
    #   Strassen on 2x2x2 blocks for j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    #   Strassen on 2x2x2 blocks for j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    #   Strassen on 2x2x2 blocks for j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    #   Strassen on 2x2x2 blocks for j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    #   Rank 1 blocks for k=4, j in {0,1}
    add_rank1_block_1x2x1(0, 0, 1, 4)
    #   Rank 1 blocks for k=4, j in {2,3}
    add_rank1_block_1x2x1(0, 2, 3, 4)
    #   Repeat for i=1
    add_strassen_block(1, 0, 0)
    add_strassen_block(1, 0, 2)
    add_strassen_block(1, 2, 0)
    add_strassen_block(1, 2, 2)
    add_rank1_block_1x2x1(1, 0, 1, 4)
    add_rank1_block_1x2x1(1, 2, 3, 4)

    # For i=2:
    #   Rank 2 blocks for j in {0,1}, k in {0,1}
    add_rank2_block_1x2x2(2, 0, 1, 0, 1)
    #   Rank 2 blocks for j in {0,1}, k in {2,3}
    add_rank2_block_1x2x2(2, 0, 1, 2, 3)
    #   Rank 2 blocks for j in {2,3}, k in {0,1}
    add_rank2_block_1x2x2(2, 2, 3, 0, 1)
    #   Rank 2 blocks for j in {2,3}, k in {2,3}
    add_rank2_block_1x2x2(2, 2, 3, 2, 3)
    #   Rank 1 blocks for k=4, j in {0,1}
    add_rank1_block_1x2x1(2, 0, 1, 4)
    #   Rank 1 blocks for k=4, j in {2,3}
    add_rank1_block_1x2x1(2, 2, 3, 4)

    # Convert to row-major format
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
