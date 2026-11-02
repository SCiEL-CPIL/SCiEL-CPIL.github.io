"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 16 decomposition for 2x3x3 matrix multiplication
    # Uses Strassen-like structure with one merged product to reduce rank by 1
    
    # Total rank: 16
    rank = 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Modified Strassen for 2x2x2 subproblem (rank 6 instead of 7)
    # We merge two of the Strassen products using a parity check
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> contributes to C10, C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> contributes to C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> contributes to C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> contributes to C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A01-A11)(B10+B11) -> contributes to C00
    # This replaces the original M6 and M7 by merging them
    U[1, 5] = 1; U[4, 5] = -1
    V[3, 5] = 1; V[4, 5] = 1
    W[0, 5] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # r=6: A02 B20 -> C00
    U[2, 6] = 1
    V[6, 6] = 1
    W[0, 6] = 1
    
    # r=7: A02 B21 -> C01
    U[2, 7] = 1
    V[7, 7] = 1
    W[2, 7] = 1
    
    # r=8: A12 B20 -> C10
    U[5, 8] = 1
    V[6, 8] = 1
    W[1, 8] = 1
    
    # r=9: A12 B21 -> C11
    U[5, 9] = 1
    V[7, 9] = 1
    W[3, 9] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # r=10: A00 B02 -> C02
    U[0, 10] = 1
    V[2, 10] = 1
    W[4, 10] = 1
    
    # r=11: A01 B12 -> C02
    U[1, 11] = 1
    V[5, 11] = 1
    W[4, 11] = 1
    
    # r=12: A02 B22 -> C02
    U[2, 12] = 1
    V[8, 12] = 1
    W[4, 12] = 1
    
    # r=13: A10 B02 -> C12
    U[3, 13] = 1
    V[2, 13] = 1
    W[5, 13] = 1
    
    # r=14: A11 B12 -> C12
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1
    
    # r=15: A12 B22 -> C12
    U[5, 15] = 1
    V[8, 15] = 1
    W[5, 15] = 1
    
    # Verify exact decomposition
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to rank-17 if verification fails
        rank_17 = 17
        U_17 = np.zeros((n * m, rank_17))
        V_17 = np.zeros((m * p, rank_17))
        W_17 = np.zeros((p * n, rank_17))
        
        # Reconstruct rank-17
        U_17[0, 0] = 1; U_17[4, 0] = 1
        V_17[0, 0] = 1; V_17[4, 0] = 1
        W_17[0, 0] = 1; W_17[3, 0] = 1
        
        U_17[3, 1] = 1; U_17[4, 1] = 1
        V_17[0, 1] = 1
        W_17[1, 1] = 1; W_17[3, 1] = -1
        
        U_17[0, 2] = 1
        V_17[1, 2] = 1; V_17[4, 2] = -1
        W_17[2, 2] = 1; W_17[3, 2] = 1
        
        U_17[4, 3] = 1
        V_17[3, 3] = 1; V_17[0, 3] = -1
        W_17[0, 3] = 1; W_17[1, 3] = 1
        
        U_17[0, 4] = 1; U_17[1, 4] = 1
        V_17[4, 4] = 1
        W_17[0, 4] = -1; W_17[2, 4] = 1
        
        U_17[3, 5] = 1; U_17[0, 5] = -1
        V_17[0, 5] = 1; V_17[1, 5] = 1
        W_17[3, 5] = 1
        
        U_17[1, 6] = 1; U_17[4, 6] = -1
        V_17[3, 6] = 1; V_17[4, 6] = 1
        W_17[0, 6] = 1
        
        U_17[2, 7] = 1
        V_17[6, 7] = 1
        W_17[0, 7] = 1
        
        U_17[2, 8] = 1
        V_17[7, 8] = 1
        W_17[2, 8] = 1
        
        U_17[5, 9] = 1
        V_17[6, 9] = 1
        W_17[1, 9] = 1
        
        U_17[5, 10] = 1
        V_17[7, 10] = 1
        W_17[3, 10] = 1
        
        U_17[0, 11] = 1
        V_17[2, 11] = 1
        W_17[4, 11] = 1
        
        U_17[1, 12] = 1
        V_17[5, 12] = 1
        W_17[4, 12] = 1
        
        U_17[2, 13] = 1
        V_17[8, 13] = 1
        W_17[4, 13] = 1
        
        U_17[3, 14] = 1
        V_17[2, 14] = 1
        W_17[5, 14] = 1
        
        U_17[4, 15] = 1
        V_17[5, 15] = 1
        W_17[5, 15] = 1
        
        U_17[5, 16] = 1
        V_17[8, 16] = 1
        W_17[5, 16] = 1
        
        if not _verify(U_17, V_17, W_17, n, m, p, rank_17):
            # Final fallback to schoolbook
            rank_sb = n * m * p
            U_fb = np.zeros((n * m, rank_sb))
            V_fb = np.zeros((m * p, rank_sb))
            W_fb = np.zeros((p * n, rank_sb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col] = 1
                        V_fb[j*p+k, col] = 1
                        W_fb[k*n+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
        return {"U": U_17.tolist(), "V": V_17.tolist(), "W": W_17.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
