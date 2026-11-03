"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product structure
    - 4x4x6 decomposed into 4x4x4 and 4x4x2 parts
    - 4x4x4 uses Strassen (rank 49)
    - 4x4x2 uses Strassen(2x2x2) x Schoolbook(2x2x1) (rank 28)
    - Total rank = 49 + 28 = 77
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Part 1: 4x4x4 multiplication
    # U1, V1, W1 for 4x4x4 (rank 49)
    U1 = np.zeros((16, 49), dtype=int)
    V1 = np.zeros((16, 49), dtype=int)
    W1 = np.zeros((16, 49), dtype=int)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U_s, V_s, W_s are 4x7
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[3,0] = 1
    
    # M2 = (a10+a11)b00
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[1,1] = 1; W_s[3,1] = -1
    
    # M3 = a00(b01-b11)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[2,2] = 1; W_s[3,2] = 1
    
    # M4 = a11(b10-b00)
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = 1; W_s[1,3] = 1
    
    # M5 = (a00+a01)b11
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = -1; W_s[2,4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = 1
    
    # Build 4x4x4 from two Strassen layers (Kronecker product of 2x2x2 with itself)
    # Index mapping for 4x4x4:
    # A[i,j] = A1[i1,j1] * A2[i2,j2]
    # U_full[i*4+j, c1*7+c2] = U_s[i1*2+j1, c1] * U_s[i2*2+j2, c2]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U_s[i1*2+j1, c1]
                            u2 = U_s[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U1[row, c] = u1 * u2
                            
                            v1 = V_s[j1*2+k1, c1] # Wait, V_s index is j*p+k? No, for 2x2x2, V is 2x2.
                            # V_s row index: j*2+k.
                            # For 4x4x4, V row index: j*4+k = (j1*2+j2)*4 + (k1*2+k2)
                            # V_full[j*4+k, c] = V_s[j1*2+k1, c1] * V_s[j2*2+k2, c2]
                            
                            w1 = W_s[k1*2+i1, c1]
                            # W_s row index: k*2+i.
                            # For 4x4x4, W row index: k*4+i = (k1*2+k2)*4 + (i1*2+i2)
                            # W_full[k*4+i, c] = W_s[k1*2+i1, c1] * W_s[k2*2+i2, c2]
            
            # Correctly build V and W for 4x4x4
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v1 = V_s[j1*2+k1, c1]
                            v2 = V_s[j2*2+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2 # Wait, 4x4x4 V is 16x49.
                                # V row index: j*4+k.
                                # j = j1*2+j2, k = k1*2+k2.
                                # row = (j1*2+j2)*4 + (k1*2+k2)
                                row = (j1*2+j2)*4 + (k1*2+k2)
                                V1[row, c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W_s[k1*2+i1, c1]
                            w2 = W_s[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                # W row index: k*4+i.
                                # k = k1*2+k2, i = i1*2+i2.
                                row = (k1*2+k2)*4 + (i1*2+i2)
                                W1[row, c] = w1 * w2

    # Part 2: 4x4x2 multiplication
    # Rank 28
    # U2, V2, W2 for 4x4x2
    U2 = np.zeros((16, 28), dtype=int)
    V2 = np.zeros((8, 28), dtype=int) # 4x2 -> 8 entries
    W2 = np.zeros((8, 28), dtype=int) # 2x4 -> 8 entries
    
    # Decompose 4x4x2 as M(2,2,2) x M(2,2,1)
    # M(2,2,2) is Strassen (rank 7)
    # M(2,2,1) is Schoolbook (rank 4)
    
    # U_s (4x7) from above
    # U_t for 2x2x1 (4x4)
    U_t = np.zeros((4, 4), dtype=int)
    V_t = np.zeros((2, 4), dtype=int) # 2x1 -> 2 entries? No, V is m*p. m=2, p=1.
    W_t = np.zeros((2, 4), dtype=int) # 1x2 -> 2 entries? No, W is p*n. p=1, n=2.
    
    # M(2,2,1): A(2x2) * B(2x1) = C(2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Terms:
    # 0: A00 B0 -> C0
    U_t[0,0] = 1; V_t[0,0] = 1; W_t[0,0] = 1
    # 1: A01 B1 -> C0
    U_t[1,1] = 1; V_t[1,1] = 1; W_t[0,1] = 1
    # 2: A10 B0 -> C1
    U_t[2,2] = 1; V_t[0,2] = 1; W_t[1,2] = 1
    # 3: A11 B1 -> C1
    U_t[3,3] = 1; V_t[1,3] = 1; W_t[1,3] = 1
    
    # Kronecker Product M(2,2,2) x M(2,2,1)
    # A(4x4) = A_s(2x2) x A_t(2x2)
    # B(4x2) = B_s(2x2) x B_t(2x1)
    # C(4x2) = C_s(2x2) x C_t(2x1)
    
    # Indices:
    # A[i,j] = A_s[i1,j1] * A_t[i2,j2]
    # B[j,k] = B_s[j1,k1] * B_t[j2,k2]
    # C[i,k] = C_s[i1,k1] * C_t[i2,k2]
    
    # U_full[i*4+j, c1*4+c2] = U_s[i1*2+j1, c1] * U_t[i2*2+j2, c2]
    # V_full[j*2+k, c1*4+c2] = V_s[j1*2+k1, c1] * V_t[j2*1+k2, c2]
    # W_full[k*4+i, c1*4+c2] = W_s[k1*2+i1, c1] * W_t[k1*1+i2, c2]
    
    for c1 in range(7):
        for c2 in range(4):
            c = c1 * 4 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U_s[i1*2+j1, c1]
                            u2 = U_t[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = (i1*2+i2)*4 + (j1*2+j2)
                                U2[row, c] = u1 * u2
                            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        # k2 is 0 only (B_t is 2x1, so k2=0)
                        v1 = V_s[j1*2+k1, c1]
                        v2 = V_t[j2*1+0, c2] # V_t row j2*1+k2. k2=0.
                        if v1 != 0 and v2 != 0:
                            row = (j1*2+j2)*2 + k1 # B is 4x2. j*2+k.
                            V2[row, c] = v1 * v2
                            
            for k1 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        w1 = W_s[k1*2+i1, c1]
                        w2 = W_t[k1*1+0, c2] # W_t row k1*1+i2. i2 is 0?
                        # C_t is 2x1. C_t[i2, 0].
                        # W_t index is k*n. k=1, n=2.
                        # W_t[k, i]. k=0 (from B_t col), i=i2.
                        # Wait.
                        # B_t is 2x1. Index (j2, k2). k2=0.
                        # C_t is 2x1. Index (i2, k2). k2=0.
                        # W_t is p*n. p=1, n=2.
                        # W_t[k2, i2].
                        # So W_t[0, i2].
                        # In the loop, c2 corresponds to a term in M(2,2,1).
                        # W_t[0, i2] is non-zero for specific i2.
                        
                        # Let's re-verify W_t.
                        # W_t is 2x4. Rows 0,1.
                        # W_t[0,0]=1 (C0), W_t[0,1]=1 (C0)
                        # W_t[1,2]=1 (C1), W_t[1,3]=1 (C1)
                        # So for a given c2, W_t[0, i2] is 1 if i2 matches.
                        
                        w2_val = W_t[0, c2] # Wait, W_t row is k2=0. Col is c2.
                        # No, W_t is 2x4.
                        # W_t[k, i]. k=0, i=i2.
                        # So we need W_t[0, i2]? No, W_t is indexed by [k*n+i, c].
                        # W_t shape (2, 4).
                        # Row index k*n+i. k=0, n=2. Row = i.
                        # So W_t[i2, c2].
                        
                        w2_val = W_t[i2, c2]
                        
                        if w1 != 0 and w2_val != 0:
                            row = (k1*1+0)*4 + (i1*2+i2) # C is 4x2. k*4+i.
                            # k = k1*1 + k2(0).
                            # i = i1*2 + i2.
                            W2[row, c] = w1 * w2_val

    # Combine Part 1 and Part 2
    # U = [U1, U2]
    # V = [V1, V2]
    # W = [W1, W2]
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
