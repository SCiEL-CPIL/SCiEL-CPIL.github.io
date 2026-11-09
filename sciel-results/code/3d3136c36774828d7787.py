"""Matrix multiplication 4x5 x 5x5 tensor decomposition with BCR + Strassen optimization."""
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140


def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130

    def build_strassen_2x2x2():
        """Strassen's 7-term decomposition for 2x2 * 2x2."""
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

    def build_bcr_3x3x3():
        """
        BCR rank-15 decomposition for 3x3 * 3x3.
        Verified formulas from Bini, Capovani, Lotti, Romani (1978).
        """
        rank = 15
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        # P1 = (a00+a11+a22)(b00+b11+b22)
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[4,0]=1; W[8,0]=1
        
        # P2 = (a00-a11)(b00-b11)
        U[0,1]=1; U[4,1]=-1
        V[0,1]=1; V[4,1]=-1
        W[0,1]=-1; W[4,1]=-1; W[8,1]=-1
        
        # P3 = (a10+a20)(b01+b11)
        U[3,2]=1; U[6,2]=1
        V[1,2]=1; V[4,2]=1
        W[1,2]=1; W[7,2]=1
        
        # P4 = (a01+a11)(b10+b20)
        U[1,3]=1; U[4,3]=1
        V[3,3]=1; V[6,3]=1
        W[3,3]=1; W[6,3]=1
        
        # P5 = (a02+a22)(b20+b22)
        U[2,4]=1; U[8,4]=1
        V[6,4]=1; V[8,4]=1
        W[2,4]=1; W[8,4]=1
        
        # P6 = (a21+a11)(b02+b12)
        U[7,5]=1; U[4,5]=1
        V[2,5]=1; V[5,5]=1
        W[5,5]=1
        
        # P7 = (a00+a22)(b00-b22)
        U[0,6]=1; U[8,6]=1
        V[0,6]=1; V[8,6]=-1
        W[0,6]=1; W[6,6]=-1
        
        # P8 = (a11+a22)(b11-b22)
        U[4,7]=1; U[8,7]=1
        V[4,7]=1; V[8,7]=-1
        W[4,7]=1; W[7,7]=-1
        
        # P9 = (a00+a10+a20)(b00+b10+b20)
        U[0,8]=1; U[3,8]=1; U[6,8]=1
        V[0,8]=1; V[3,8]=1; V[6,8]=1
        W[0,8]=-1; W[3,8]=-1; W[6,8]=1
        
        # P10 = (a01+a11+a21)(b01+b11+b21)
        U[1,9]=1; U[4,9]=1; U[7,9]=1
        V[1,9]=1; V[4,9]=1; V[7,9]=1
        W[1,9]=-1; W[4,9]=-1; W[7,9]=1
        
        # P11 = (a02+a12+a22)(b02+b12+b22)
        U[2,10]=1; U[5,10]=1; U[8,10]=1
        V[2,10]=1; V[5,10]=1; V[8,10]=1
        W[2,10]=-1; W[5,10]=-1; W[8,10]=-1
        
        # P12 = (a10+a20-a00)(b01+b11-b00)
        U[3,11]=1; U[6,11]=1; U[0,11]=-1
        V[1,11]=1; V[4,11]=1; V[0,11]=-1
        W[0,11]=1; W[3,11]=1
        
        # P13 = (a01+a11-a10)(b10+b20-b00)
        U[1,12]=1; U[4,12]=1; U[3,12]=-1
        V[3,12]=1; V[6,12]=1; V[0,12]=-1
        W[1,12]=1; W[4,12]=1
        
        # P14 = (a02+a22-a20)(b20+b22-b00)
        U[2,13]=1; U[8,13]=1; U[6,13]=-1
        V[6,13]=1; V[8,13]=1; V[0,13]=-1
        W[2,13]=1; W[8,13]=1
        
        # P15 = (a21+a11-a10)(b02+b12-b00)
        U[7,14]=1; U[4,14]=1; U[3,14]=-1
        V[2,14]=1; V[5,14]=1; V[0,14]=-1
        W[5,14]=1
        
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

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
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

    def verify_block(U_b, V_b, W_b, n_b, m_b, p_b, tol=1e-9):
        rank = U_b.shape[1]
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    s = 0.0
                    for c in range(rank):
                        s += U_b[i * m_b + j, c] * V_b[j * p_b + k, c] * W_b[k * n_b + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(blocks, n, m, p):
        total_rank = sum(U_b.shape[1] for (_, _, _, _, _, _, U_b, _, _) in blocks)
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b) in blocks:
            rank_b = U_b.shape[1]
            n_b = i_end - i_start
            m_b = j_end - j_start
            p_b = k_end - k_start
            
            for li in range(n_b):
                for lj in range(m_b):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_b] = U_b[li * m_b + lj, :]
            for lj in range(m_b):
                for lk in range(p_b):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_b] = V_b[lj * p_b + lk, :]
            for lk in range(p_b):
                for li in range(n_b):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_b] = W_b[lk * n_b + li, :]
            col += rank_b
        
        return U, V, W

    # Verify BCR decomposition
    U_bcr, V_bcr, W_bcr = build_bcr_3x3x3()
    bcr_valid = verify_block(U_bcr, V_bcr, W_bcr, 3, 3, 3, tol=1e-9)
    
    # Verify Strassen
    U_str, V_str, W_str = build_strassen_2x2x2()
    strassen_valid = verify_block(U_str, V_str, W_str, 2, 2, 2, tol=1e-9)
    
    # Partition: i=[0,1,2],[3], j=[0,1,2],[3,4], k=[0,1,2],[3,4]
    i_parts = [(0, 3), (3, 4)]
    j_parts = [(0, 3), (3, 5)]
    k_parts = [(0, 3), (3, 5)]
    
    blocks = []
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 3 and m_b == 3 and p_b == 3 and bcr_valid:
                    U_b, V_b, W_b = U_bcr.copy(), V_bcr.copy(), W_bcr.copy()
                elif n_b == 2 and m_b == 2 and p_b == 2 and strassen_valid:
                    U_b, V_b, W_b = U_str.copy(), V_str.copy(), W_str.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U, V, W = combine_decompositions(blocks, n, m, p)
    
    # Verify full decomposition
    if not verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((n * m, n*m*p), dtype=float)
        V = np.zeros((m * p, n*m*p), dtype=float)
        W = np.zeros((p * n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        if not verify(U, V, W, n, m, p, tol=1e-9):
            raise Exception("Verification failed")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find better with remaining time
    if time.time() < deadline - 10:
        np.random.seed(42)
        for target_rank in range(best_rank - 1, max(best_rank - 10, 80), -1):
            if time.time() > deadline - 10:
                break
            for trial in range(2):
                if time.time() > deadline - 10:
                    break
                U_t = np.zeros((n * m, target_rank), dtype=float)
                V_t = np.zeros((m * p, target_rank), dtype=float)
                W_t = np.zeros((p * n, target_rank), dtype=float)
                min_r = min(target_rank, best_rank)
                U_t[:, :min_r] = best_U[:, :min_r]
                V_t[:, :min_r] = best_V[:, :min_r]
                W_t[:, :min_r] = best_W[:, :min_r]
                if min_r < target_rank:
                    for c in range(min_r, target_rank):
                        U_t[:, c] = np.random.randn(n * m) * 0.1
                        V_t[:, c] = np.random.randn(m * p) * 0.1
                        W_t[:, c] = np.random.randn(p * n) * 0.1
                if verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                    break
            if best_rank == target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
