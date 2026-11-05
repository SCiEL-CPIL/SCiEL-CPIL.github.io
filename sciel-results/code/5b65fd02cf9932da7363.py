"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin

    # We construct a rank 100 decomposition using block Strassen.
    # Split A (4x5) into A1 (2x5) and A2 (2x5).
    # Split B (5x6) into B1 (5x3) and B2 (5x3).
    # For each product Ak * Bl (2x5x3), split inner dim 5 into 2+3.
    # Ak * Bl = Ak_2 * Bl_2 (2x2x2) + Ak_3 * Bl_3 (2x3x3).
    # Rank(2x2x2) = 7 (Strassen).
    # Rank(2x3x3) = 18 (Schoolbook).
    # Total Rank = 4 * (7 + 18) = 100.

    # Strassen factors for 2x2x2
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need U, V, W such that sum U*V*W = T.
    # For 2x2x2, U is 4x7, V is 4x7, W is 4x7.
    
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        # M1: (A00+A11)(B00+B11)
        # U[0,0]=1, U[3,0]=1 (A00, A11)
        # V[0,0]=1, V[3,0]=1 (B00, B11) -> V[j*2+k]
        # B00 is j=0,k=0 -> idx 0. B11 is j=1,k=1 -> idx 3.
        # W[0,0]=1, W[3,0]=1 (C00, C11) -> W[k*2+i]
        # C00 is k=0,i=0 -> idx 0. C11 is k=1,i=1 -> idx 3.
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2: (A10+A11)B00
        # A10 is i=1,j=0 -> idx 2. A11 is i=1,j=1 -> idx 3.
        # B00 is j=0,k=0 -> idx 0.
        # C10 is k=0,i=1 -> idx 2. C11 is k=1,i=1 -> idx 3.
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = 1
        
        # M3: A00(B01-B11)
        # A00 is i=0,j=0 -> idx 0.
        # B01 is j=0,k=1 -> idx 1. B11 is j=1,k=1 -> idx 3.
        # C01 is k=1,i=0 -> idx 1. C11 is k=1,i=1 -> idx 3.
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # M4: A11(B10-B00)
        # A11 is i=1,j=1 -> idx 3.
        # B10 is j=1,k=0 -> idx 2. B00 is j=0,k=0 -> idx 0.
        # C10 is k=0,i=1 -> idx 2. C00 is k=0,i=0 -> idx 0.
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[2,3] = 1; W[0,3] = 1
        
        # M5: (A00+A01)B11
        # A00 is i=0,j=0 -> idx 0. A01 is i=0,j=1 -> idx 1.
        # B11 is j=1,k=1 -> idx 3.
        # C01 is k=1,i=0 -> idx 1. C11 is k=1,i=1 -> idx 3.
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[3,4] = 1
        
        # M6: (A10-A00)(B00+B01)
        # A10 is i=1,j=0 -> idx 2. A00 is i=0,j=0 -> idx 0.
        # B00 is j=0,k=0 -> idx 0. B01 is j=0,k=1 -> idx 1.
        # C10 is k=0,i=1 -> idx 2. C00 is k=0,i=0 -> idx 0.
        # C11 is k=1,i=1 -> idx 3. C01 is k=1,i=0 -> idx 1.
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[2,5] = 1; W[0,5] = 1
        W[3,5] = 1; W[1,5] = 1
        
        # M7: (A01-A11)(B10+B11)
        # A01 is i=0,j=1 -> idx 1. A11 is i=1,j=1 -> idx 3.
        # B10 is j=1,k=0 -> idx 2. B11 is j=1,k=1 -> idx 3.
        # C00 is k=0,i=0 -> idx 0. C10 is k=0,i=1 -> idx 2.
        # C01 is k=1,i=0 -> idx 1. C11 is k=1,i=1 -> idx 3.
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[2,6] = 1
        W[1,6] = 1; W[3,6] = 1
        
        return U, V, W

    def get_schoolbook_233():
        U = np.zeros((6, 18))
        V = np.zeros((9, 18))
        W = np.zeros((6, 18))
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = get_strassen_222()
    U_b, V_b, W_b = get_schoolbook_233()

    # Global Dimensions
    N, M, P = 4, 5, 6
    Rank = 100
    
    U_global = np.zeros((N*M, Rank))
    V_global = np.zeros((M*P, Rank))
    W_global = np.zeros((P*N, Rank))
    
    rank_offset = 0
    
    # 4 blocks: k_block in {0,1}, l_block in {0,1}
    for k_block in range(2): # A rows
        for l_block in range(2): # B cols
            
            # Part 1: 2x2x2 (Inner 2)
            # Local U: 4x7, V: 4x7, W: 4x7
            # Map to global
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        if U_s[i*2+j, r] != 0:
                            gi = k_block * 2 + i
                            gj = 0 + j # Inner 0,1
                            U_global[gi*M + gj, rank_offset + r] = U_s[i*2+j, r]
                
                # V
                for j in range(2):
                    for k in range(2):
                        if V_s[j*2+k, r] != 0:
                            gj = 0 + j
                            gk = l_block * 3 + k
                            V_global[gj*P + gk, rank_offset + r] = V_s[j*2+k, r]
                
                # W
                for k in range(2):
                    for i in range(2):
                        if W_s[k*2+i, r] != 0:
                            gk = l_block * 3 + k
                            gi = k_block * 2 + i
                            W_global[gk*N + gi, rank_offset + r] = W_s[k*2+i, r]
            
            rank_offset += 7
            
            # Part 2: 2x3x3 (Inner 3)
            # Local U: 6x18, V: 9x18, W: 6x18
            # Map to global
            for r in range(18):
                # U
                for i in range(2):
                    for j in range(3):
                        if U_b[i*3+j, r] != 0:
                            gi = k_block * 2 + i
                            gj = 2 + j # Inner 2,3,4
                            U_global[gi*M + gj, rank_offset + r] = U_b[i*3+j, r]
                
                # V
                for j in range(3):
                    for k in range(3):
                        if V_b[j*3+k, r] != 0:
                            gj = 2 + j
                            gk = l_block * 3 + k
                            V_global[gj*P + gk, rank_offset + r] = V_b[j*3+k, r]
                
                # W
                for k in range(3):
                    for i in range(2):
                        if W_b[k*2+i, r] != 0:
                            gk = l_block * 3 + k
                            gi = k_block * 2 + i
                            W_global[gk*N + gi, rank_offset + r] = W_b[k*2+i, r]
            
            rank_offset += 18

    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not verify(U_global, V_global, W_global, N, M, P):
        # Fallback to schoolbook if construction failed
        rank_school = N * M * P
        U_fallback = np.zeros((N*M, rank_school))
        V_fallback = np.zeros((M*P, rank_school))
        W_fallback = np.zeros((P*N, rank_school))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_fallback[i*M+j, col] = 1
                    V_fallback[j*P+k, col] = 1
                    W_fallback[k*N+i, col] = 1
                    col += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
