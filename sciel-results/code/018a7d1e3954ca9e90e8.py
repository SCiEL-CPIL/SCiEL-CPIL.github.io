"""Matrix multiplication 4x5 x 5x5 tensor decomposition with Strassen optimization."""
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill


def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130  # Safety margin

    def build_strassen_2x2x2():
        """
        Build Strassen's 7-term decomposition for 2x2 * 2x2.
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # Term 2: M2 = (A10+A11)B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # Term 3: M3 = A00(B10-B11)
        U[0, 2] = 1.0
        V[2, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # Term 4: M4 = A11(B01-B10)
        U[3, 3] = 1.0
        V[1, 3] = 1.0; V[2, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # Term 5: M5 = (A00+A01)B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[2, 5] = 1.0
        W[3, 5] = 1.0
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[1, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0; W[1, 6] = 1.0
        
        return U, V, W

    def build_2x3x3_rank17():
        """
        Build rank-17 decomposition for 2x3 * 3x3 -> 2x3.
        Partition: Strassen for (i in {0,1}, j in {0,1}, k in {0,1}) [rank 7]
                  + schoolbook for remaining 10 entries [rank 10]
        Total: 17
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        rank_total = 17
        U = np.zeros((6, rank_total), dtype=float)  # 2*3 = 6
        V = np.zeros((9, rank_total), dtype=float)  # 3*3 = 9
        W = np.zeros((6, rank_total), dtype=float)  # 3*2 = 6
        
        # Place Strassen for j in {0,1}, k in {0,1}
        # U: local (i, j) where i in {0,1}, j in {0,1} -> global U[i*3+j, :]
        # V: local (j, k) where j in {0,1}, k in {0,1} -> global V[j*3+k, :]
        # W: local (k, i) where k in {0,1}, i in {0,1} -> global W[k*2+i, :]
        for li in range(2):
            for lj in range(2):
                U[li * 3 + lj, 0:7] = U_s[li * 2 + lj, :]
        for lj in range(2):
            for lk in range(2):
                V[lj * 3 + lk, 0:7] = V_s[lj * 2 + lk, :]
        for lk in range(2):
            for li in range(2):
                W[lk * 2 + li, 0:7] = W_s[lk * 2 + li, :]
        
        # Schoolbook for remaining entries (rank 10)
        # Remaining (i,j,k) tuples where j==2 or k==2:
        # (i, j, k) for i in {0,1}:
        # j in {0,1}, k=2: 4 entries
        # j=2, k in {0,1}: 4 entries  
        # j=2, k=2: 2 entries
        remaining = []
        for i in range(2):
            for j in range(2):
                remaining.append((i, j, 2))
            for k in range(2):
                remaining.append((i, 2, k))
            remaining.append((i, 2, 2))
        
        for idx, (i, j, k) in enumerate(remaining):
            col = 7 + idx
            U[i * 3 + j, col] = 1.0
            V[j * 3 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
        
        return U, V, W

    def build_schoolbook(n, m, p):
        """Build schoolbook decomposition."""
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify tensor decomposition."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(blocks, n, m, p):
        """
        Combine block decompositions into full decomposition.
        blocks: list of (i_start, j_start, k_start, i_end, j_end, k_end, U_block, V_block, W_block)
        """
        total_rank = sum(len(block) for block in blocks)
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_block, V_block, W_block) in blocks:
            rank_block = U_block.shape[1]
            for li in range(i_end - i_start):
                for lj in range(j_end - j_start):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_block] = U_block[li * (j_end-j_start) + lj, :]
            for lj in range(j_end - j_start):
                for lk in range(k_end - k_start):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_block] = V_block[lj * (k_end-k_start) + lk, :]
            for lk in range(k_end - k_start):
                for li in range(i_end - i_start):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_block] = W_block[lk * (i_end-i_start) + li, :]
            col += rank_block
        
        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        rank_b = n_b * m_b * p_b
        U_b = np.zeros((n_b * m_b, rank_b), dtype=float)
        V_b = np.zeros((m_b * p_b, rank_b), dtype=float)
        W_b = np.zeros((p_b * n_b, rank_b), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, col] = 1.0
                    V_b[j * p_b + k, col] = 1.0
                    W_b[k * n_b + i, col] = 1.0
                    col += 1
        return U_b, V_b, W_b

    # Build the decomposition using Strassen for 2x2x2 and 2x3x3 rank-17 blocks
    # Partition: i into [0,1],[2,3]; j into [0,1],[2,3],[4]; k into [0,1],[2,3],[4]
    
    U_s, V_s, W_s = build_strassen_2x2x2()
    U_233, V_233, W_233 = build_2x3x3_rank17()
    
    blocks = []
    
    i_parts = [(0, 1), (2, 3)]
    j_parts = [(0, 1), (2, 3), (4, 4)]
    k_parts = [(0, 1), (2, 3), (4, 4)]
    
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    U_b, V_b, W_b = U_s.copy(), V_s.copy(), W_s.copy()
                elif n_b == 2 and m_b == 3 and p_b == 3:
                    U_b, V_b, W_b = U_233.copy(), V_233.copy(), W_233.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U, V, W = combine_decompositions(blocks, n, m, p)
    
    # Verify
    if not verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback: try the simpler partition
        blocks2 = []
        for i_start, i_end in i_parts:
            for j_start, j_end in j_parts:
                for k_start, k_end in k_parts:
                    n_b = i_end - i_start
                    m_b = j_end - j_start
                    p_b = k_end - k_start
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        U_b, V_b, W_b = U_s.copy(), V_s.copy(), W_s.copy()
                    else:
                        U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                    blocks2.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
        U, V, W = combine_decompositions(blocks2, n, m, p)
        if not verify(U, V, W, n, m, p, tol=1e-9):
            U, V, W = build_schoolbook(n, m, p)
            if not verify(U, V, W, n, m, p, tol=1e-9):
                raise Exception("Verification failed")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
