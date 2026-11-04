"""Deterministic rank-reduction for 4x5 * 5x6 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 decomposition
    # U has shape (2,2,7), V has shape (2,2,7), W has shape (2,2,7)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    
    def strassen_2x2():
        """Return Strassen decomposition for 2x2x2 tensor."""
        U = np.zeros((2, 2, 7), dtype=np.int64)
        V = np.zeros((2, 2, 7), dtype=np.int64)
        W = np.zeros((2, 2, 7), dtype=np.int64)
        
        # c=0: U[0,0]=1, V[1,0]=1, W[1,0]=1 -> contributes to (0,0,1)
        U[0, 0, 0] = 1
        V[1, 0, 0] = 1
        W[1, 0, 0] = 1
        
        # c=1: U[1,1]=1, V[0,1]=1, W[0,1]=1 -> contributes to (1,1,0)
        U[1, 1, 1] = 1
        V[0, 1, 1] = 1
        W[0, 1, 1] = 1
        
        # c=2: U[1,0]=1, V[0,1]=1, W[0,0]=1 -> contributes to (1,0,0)
        U[1, 0, 2] = 1
        V[0, 1, 2] = 1
        W[0, 0, 2] = 1
        
        # c=3: U[0,1]=1, V[1,0]=1, W[1,1]=1 -> contributes to (0,1,1)
        U[0, 1, 3] = 1
        V[1, 0, 3] = 1
        W[1, 1, 3] = 1
        
        # c=4: U[0,0]=1, U[0,1]=1, U[1,0]=1, U[1,1]=1
        #      V[0,0]=1, V[1,1]=1
        #      W[0,0]=1, W[1,1]=1
        # This is the "sum" pattern: M1 = (A00+A11)(B00+B11)
        # Actually let me use the standard Strassen formulas
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 + M6
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let me redefine properly
        # For 2x2 matmul: C = A*B where A,B,C are 2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # The tensor T[i,j,k] where i indexes row of A (and C), j indexes col of A (and row of B), k indexes col of B (and C)
        # So T[i,j,k] = 1 means: the (i,k) entry of C gets contribution from A[i,j]*B[j,k]
        
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 + M6
        
        U2 = np.zeros((2, 2, 7), dtype=np.int64)
        V2 = np.zeros((2, 2, 7), dtype=np.int64)
        W2 = np.zeros((2, 2, 7), dtype=np.int64)
        
        # M1 = (A00+A11)*(B00+B11)
        # U for M1: A00 and A11 have coefficient 1
        U2[0, 0, 0] = 1  # A00
        U2[1, 1, 0] = 1  # A11
        # V for M1: B00 and B11 have coefficient 1
        V2[0, 0, 0] = 1  # B00
        V2[1, 1, 0] = 1  # B11
        # W for M1: contributes to C00 (+1), C10 (+1), C11 (+1)
        W2[0, 0, 0] = 1  # C00
        W2[0, 1, 0] = 1  # C10 (k=0, i=1)
        W2[1, 1, 0] = 1  # C11 (k=1, i=1)
        
        # M2 = (A10+A11)*B00
        U2[1, 0, 1] = 1  # A10
        U2[1, 1, 1] = 1  # A11
        V2[0, 0, 1] = 1  # B00
        # W for M2: contributes to C10 (+1)
        W2[0, 1, 1] = 1  # C10
        
        # M3 = A00*(B10-B11)
        U2[0, 0, 2] = 1  # A00
        V2[1, 0, 2] = 1  # B10
        V2[1, 1, 2] = -1 # B11
        # W for M3: contributes to C01 (+1), C11 (-1)
        W2[1, 0, 2] = 1  # C01 (k=1, i=0)
        W2[1, 1, 2] = -1 # C11 (k=1, i=1)
        
        # M4 = A11*(B01-B10)
        U2[1, 1, 3] = 1  # A11
        V2[0, 1, 3] = 1  # B01
        V2[1, 0, 3] = -1 # B10
        # W for M4: contributes to C00 (-1), C01 (+1)
        W2[0, 0, 3] = -1 # C00
        W2[1, 0, 3] = 1  # C01
        
        # M5 = (A00+A01)*B11
        U2[0, 0, 4] = 1  # A00
        U2[0, 1, 4] = 1  # A01
        V2[1, 1, 4] = 1  # B11
        # W for M5: contributes to C00 (+1), C11 (+1)
        W2[0, 0, 4] = 1  # C00
        W2[1, 1, 4] = 1  # C11
        
        # M6 = (A10-A00)*(B00+B01)
        U2[1, 0, 5] = 1  # A10
        U2[0, 0, 5] = -1 # A00
        V2[0, 0, 5] = 1  # B00
        V2[0, 1, 5] = 1  # B01
        # W for M6: contributes to C11 (+1)
        W2[1, 1, 5] = 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U2[0, 1, 6] = 1  # A01
        U2[1, 1, 6] = -1 # A11
        V2[1, 0, 6] = 1  # B10
        V2[1, 1, 6] = 1  # B11
        # W for M7: contributes to C00 (-1)
        W2[0, 0, 6] = -1 # C00
        
        # Verify
        T = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
        if not np.allclose(T, np.ones((2, 2, 2)), atol=1e-9):
            # Fallback to schoolbook for 2x2
            U2 = np.zeros((2, 2, 4), dtype=np.int64)
            V2 = np.zeros((2, 2, 4), dtype=np.int64)
            W2 = np.zeros((2, 2, 4), dtype=np.int64)
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U2[i, j, col] = 1
                        V2[j, k, col] = 1
                        W2[k, i, col] = 1
                        col += 1
            return U2, V2, W2
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    rank_2 = U2.shape[2]
    
    # Now build 4x5x6 decomposition
    # Strategy: Use Kronecker product of Strassen for the 4x4x4 core (i in 0..3, j in 0..3, k in 0..3)
    # Then add schoolbook terms for the remaining indices
    
    # Kronecker product: U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    # This gives a 4x4x4 decomposition with rank rank_2 * rank_2 = 49
    
    # U2 has shape (2, 2, rank_2), so U4 will have shape (4, 4, rank_2*rank_2)
    # U4[i1*2+i2, j1*2+j2, c1*rank_2+c2] = U2[i1, j1, c1] * U2[i2, j2, c2]
    
    U4 = np.zeros((4, 4, rank_2 * rank_2), dtype=np.int64)
    V4 = np.zeros((4, 4, rank_2 * rank_2), dtype=np.int64)
    W4 = np.zeros((4, 4, rank_2 * rank_2), dtype=np.int64)
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(rank_2):
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(rank_2):
                            idx = c1 * rank_2 + c2
                            val = U2[i1, j1, c1] * U2[i2, j2, c2]
                            if val != 0:
                                U4[i1 * 2 + i2, j1 * 2 + j2, idx] += val
    
    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(rank_2):
                for j2 in range(2):
                    for k2 in range(2):
                        for c2 in range(rank_2):
                            idx = c1 * rank_2 + c2
                            val = V2[j1, k1, c1] * V2[j2, k2, c2]
                            if val != 0:
                                V4[j1 * 2 + j2, k1 * 2 + k2, idx] += val
    
    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(rank_2):
                for k2 in range(2):
                    for i2 in range(2):
                        for c2 in range(rank_2):
                            idx = c1 * rank_2 + c2
                            val = W2[k1, i1, c1] * W2[k2, i2, c2]
                            if val != 0:
                                W4[k1 * 2 + k2, i1 * 2 + i2, idx] += val
    
    # Verify 4x4x4 core
    T4 = np.einsum('ijc,jkc,kic->ijk', U4, V4, W4)
    core_valid = np.allclose(T4, np.ones((4, 4, 4)), atol=1e-9)
    
    if not core_valid:
        # Fallback: use schoolbook for everything
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=np.int64)
        V = np.zeros((m * p, rank_school), dtype=np.int64)
        W = np.zeros((p * n, rank_school), dtype=np.int64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now extend to 4x5x6
    # We need to cover:
    # - i in 0..3, j in 0..3, k in 0..3 (already done, 49 terms)
    # - i in 0..3, j in 0..3, k in 4..5 (need schoolbook: 4*4*2 = 32 terms)
    # - i in 0..3, j in 4, k in 0..5 (need schoolbook: 4*1*6 = 24 terms)
    # Total additional: 32 + 24 = 56 terms
    # Total rank: 49 + 56 = 105
    
    rank_total = 49 + 56
    U_full = np.zeros((n * m, rank_total), dtype=np.int64)
    V_full = np.zeros((m * p, rank_total), dtype=np.int64)
    W_full = np.zeros((p * n, rank_total), dtype=np.int64)
    
    # Copy the 4x4x4 core (columns 0..48)
    U_full[:4*4, :49] = U4.reshape(4*4, 49)
    V_full[:4*4, :49] = V4.reshape(4*4, 49)
    W_full[:4*4, :49] = W4.reshape(4*4, 49)
    
    col = 49
    
    # Add schoolbook terms for j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_full[i * 5 + j, col] = 1
                V_full[j * 6 + k, col] = 1
                W_full[k * 4 + i, col] = 1
                col += 1
    
    # Add schoolbook terms for j=4, k in 0..5
    for i in range(4):
        for k in range(6):
            U_full[i * 5 + 4, col] = 1
            V_full[4 * 6 + k, col] = 1
            W_full[k * 4 + i, col] = 1
            col += 1
    
    # Verify the full decomposition
    U_r = U_full.reshape(n, m, rank_total)
    V_r = V_full.reshape(m, p, rank_total)
    W_r = W_full.reshape(p, n, rank_total)
    T_full = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T_full, np.ones((n, m, p)), atol=1e-9):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=np.int64)
        V = np.zeros((m * p, rank_school), dtype=np.int64)
        W = np.zeros((p * n, rank_school), dtype=np.int64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
