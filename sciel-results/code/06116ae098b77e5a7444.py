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
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
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
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    # Build 3x5x6 from 2x5x6 (Strassen for 2x2x2 sub-blocks) and 1x5x6 (schoolbook)
    # 2x5x6: split 5=2+2+1, 6=2+2+2
    #   6 blocks of 2x2x2: Strassen rank 7 each -> 42
    #   3 blocks of 2x1x2: schoolbook rank 4 each -> 12
    #   Total 2x5x6: 54
    # 1x5x6: schoolbook rank 30
    # Grand total: 84

    def compute_2x5x6():
        """Compute 2x5x6 using Strassen for 2x2x2 sub-blocks."""
        n_sub, m_sub, p_sub = 2, 5, 6
        U_parts = []
        V_parts = []
        W_parts = []
        
        for j_block in range(3):  # 0,1,2 for the 5 dimension (sizes 2,2,1)
            m_block = 2 if j_block < 2 else 1
            j_offset = j_block * 2
            
            for k_block in range(3):  # 0,1,2 for the 6 dimension (sizes 2,2,2)
                p_block = 2
                k_offset = k_block * 2
                
                if m_block == 2:
                    # 2x2x2: use Strassen
                    Us, Vs, Ws = strassen_2x2x2()
                    rank_s = Us.shape[1]
                    
                    U_block = np.zeros((n_sub * m_sub, rank_s))
                    V_block = np.zeros((m_sub * p_sub, rank_s))
                    W_block = np.zeros((p_sub * n_sub, rank_s))
                    
                    for r in range(rank_s):
                        for i in range(2):
                            for j in range(2):
                                U_block[i * m_sub + (j + j_offset), r] = Us[i * 2 + j, r]
                        for j in range(2):
                            for k in range(2):
                                V_block[(j + j_offset) * p_sub + (k + k_offset), r] = Vs[j * 2 + k, r]
                        for k in range(2):
                            for i in range(2):
                                W_block[(k + k_offset) * n_sub + i, r] = Ws[k * 2 + i, r]
                    
                    U_parts.append(U_block)
                    V_parts.append(V_block)
                    W_parts.append(W_block)
                else:
                    # 2x1x2: schoolbook, rank 4
                    rank_s = 4
                    U_block = np.zeros((n_sub * m_sub, rank_s))
                    V_block = np.zeros((m_sub * p_sub, rank_s))
                    W_block = np.zeros((p_sub * n_sub, rank_s))
                    
                    col = 0
                    for i in range(2):
                        for k in range(2):
                            U_block[i * m_sub + j_offset, col] = 1
                            V_block[j_offset * p_sub + (k + k_offset), col] = 1
                            W_block[(k + k_offset) * n_sub + i, col] = 1
                            col += 1
                    
                    U_parts.append(U_block)
                    V_parts.append(V_block)
                    W_parts.append(W_block)
        
        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)
        return U, V, W

    def compute_1x5x6():
        """Compute 1x5x6 schoolbook."""
        return schoolbook(1, 5, 6)

    U1, V1, W1 = compute_2x5x6()
    U2, V2, W2 = compute_1x5x6()
    
    rank_total = U1.shape[1] + U2.shape[1]
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Embed 2x5x6 (rows 0,1 of A)
    for r in range(U1.shape[1]):
        for i in range(2):
            for j in range(5):
                U[i * m + j, r] = U1[i * 5 + j, r]
        for j in range(5):
            for k in range(6):
                V[j * p + k, r] = V1[j * 6 + k, r]
        for k in range(6):
            for i in range(2):
                W[k * n + i, r] = W1[k * 2 + i, r]
    
    # Embed 1x5x6 (row 2 of A)
    offset = U1.shape[1]
    for r in range(U2.shape[1]):
        for j in range(5):
            U[2 * m + j, offset + r] = U2[j, r]
        for j in range(5):
            for k in range(6):
                V[j * p + k, offset + r] = V2[j * 6 + k, r]
        for k in range(6):
            W[k * n + 2, offset + r] = W2[k, r]
    
    # Verify
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
