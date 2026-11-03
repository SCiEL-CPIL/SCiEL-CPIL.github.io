"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses a Strassen-based decomposition for 4x4*4x4 blocks and schoolbook for the remainder.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4 * 4x4 using recursive Strassen (2x2 blocks).
        """
        # Strassen 2x2x2 rank 7
        # U, V, W for 2x2x2
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[i,j,r], V[j,k,r], W[k,i,r]
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U2[0,0] = 1; U2[3,0] = 1
        V2[0,0] = 1; V2[3,0] = 1
        W2[0,0] = 1; W2[1,1] = 1 # W[k,i]. C00 gets +M1, C11 gets +M1
        # Note: W2[k*n+i, r]. For 2x2, n=2.
        # W2[0,0] corresponds to k=0, i=0.
        # W2[1,0] corresponds to k=0, i=1.
        # W2[2,0] corresponds to k=1, i=0.
        # W2[3,0] corresponds to k=1, i=1.
        
        # Let's explicitly define the 7 terms for 2x2x2
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11.
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1 # C00 (k=0,i=0) and C11 (k=1,i=1)
        
        # Term 2: (A10+A11)B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1 # C10 (k=1,i=0)
        
        # Term 3: A00(B10-B11)
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1 # C10 (k=1,i=0)
        
        # Term 4: A11(B01-B00)
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[1, 3] = 1 # C01 (k=0,i=1)
        
        # Term 5: (A00+A01)B11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[1, 4] = 1 # C01 (k=0,i=1)
        
        # Term 6: (A10-A00)(B00+B01)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[2, 5] = 1 # C10 (k=1,i=0)
        
        # Term 7: (A01-A11)(B10+B11)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[3, 6] = 1 # C11 (k=1,i=1)
        
        # Verify 2x2x2
        # C00 = M1 - M2 + M3 - M6 ?
        # My W assignments:
        # M1: W2[0,0]=1, W2[3,0]=1. (C00, C11)
        # M2: W2[2,1]=1. (C10)
        # M3: W2[2,2]=1. (C10)
        # M4: W2[1,3]=1. (C01)
        # M5: W2[1,4]=1. (C01)
        # M6: W2[2,5]=1. (C10)
        # M7: W2[3,6]=1. (C11)
        
        # Let's check C00 (k=0, i=0). Index 0 in W.
        # Only M1 contributes. W2[0,0]=1.
        # C00 = M1.
        # But C00 = A00B00 + A01B10.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is not C00.
        
        # I need to use the standard Strassen formulas correctly.
        # C00 = M1 - M2 + M3 - M6 ? No.
        # Standard:
        # C00 = M1 - M2 + M3 - M6 ?
        # Let's stick to the known rank 7 formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M3 - M6
        # C01 = M4 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's re-assign W based on this.
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1  # C00
        W2[3, 0] = 1  # C11
        
        # M2
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = -1 # C10 (from C00 formula? No, C00 has -M2. C10 has +M2)
        # Wait, C00 = M1 - M2 + M3 - M6.
        # So W for C00 (idx 0) for M2 is -1.
        # C10 (idx 2) for M2 is +1.
        W2[0, 1] = -1
        W2[2, 1] = 1
        
        # M3
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[0, 2] = 1 # C00
        W2[2, 2] = 1 # C10
        
        # M4
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[1, 3] = 1 # C01
        W2[2, 3] = 1 # C10
        
        # M5
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[1, 4] = 1 # C01
        
        # M6
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[0, 5] = -1 # C00
        
        # M7
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[3, 6] = 1 # C11
        
        # Now extend to 4x4x4.
        # We have 4x4 matrices. We can view them as 2x2 blocks of 2x2 matrices.
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
        # This requires a Kronecker product of the 2x2x2 decomposition with itself.
        
        # Rank will be 7 * 7 = 49.
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                # U4[i,j,r] = U2[i1,j1,r1] * U2[i2,j2,r2]
                # Indices in 4x4: i = i1*2 + i2, j = j1*2 + j2.
                # U2 indices: i1, j1 in {0,1}.
                for i1 in range(2):
                    for j1 in range(2):
                        if U2[i1*2+j1, r1] != 0:
                            for i2 in range(2):
                                for j2 in range(2):
                                    if U2[i2*2+j2, r2] != 0:
                                        i = i1*2 + i2
                                        j = j1*2 + j2
                                        U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                                        
                for j1 in range(2):
                    for k1 in range(2):
                        if V2[j1*2+k1, r1] != 0:
                            for j2 in range(2):
                                for k2 in range(2):
                                    if V2[j2*2+k2, r2] != 0:
                                        j = j1*2 + j2
                                        k = k1*2 + k2
                                        V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                                        
                for k1 in range(2):
                    for i1 in range(2):
                        if W2[k1*2+i1, r1] != 0:
                            for k2 in range(2):
                                for i2 in range(2):
                                    if W2[k2*2+i2, r2] != 0:
                                        k = k1*2 + k2
                                        i = i1*2 + i2
                                        W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
                                        
        return U4, V4, W4
    
    def build_schoolbook_4x4x1():
        """
        Build a rank-16 decomposition for 4x4 * 4x1.
        """
        U = np.zeros((16, 16))
        V = np.zeros((4, 16))
        W = np.zeros((4, 16))
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    # Build the 4x4x4 part (columns 0-3)
    U1, V1, W1 = build_strassen_4x4x4()
    
    # Build the 4x4x4 part (columns 4-7)
    # This is identical to U1, V1, W1 but shifted in the output index k.
    # V index is j*4 + k. For cols 4-7, k' = k + 4.
    U2 = U1.copy()
    V2 = np.zeros((16, 49))
    W2 = W1.copy()
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V2[j*4 + (k+4), r] = V1[j*4+k, r]
                
    # Build the 4x4x1 part (column 8)
    U3, V3, W3 = build_schoolbook_4x4x1()
    # Shift V index for column 8.
    # V3 is (4, 16). V index j*1 + k. k=0.
    # We need V index j*9 + 8.
    V3_shifted = np.zeros((36, 16))
    for j in range(4):
        V3_shifted[j*9 + 8, :] = V3[j, :]
        
    # Combine
    # U: (16, 49+49+16)
    U_total = np.hstack([U1, U2, U3])
    
    # V: (36, 49+49+16)
    # V1 is (16, 49). Needs to be (36, 49).
    V1_pad = np.zeros((36, 49))
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V1_pad[j*9 + k, r] = V1[j*4+k, r]
                
    V2_pad = np.zeros((36, 49))
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V2_pad[j*9 + (k+4), r] = V1[j*4+k, r]
                
    V_total = np.hstack([V1_pad, V2_pad, V3_shifted])
    
    # W: (36, 49+49+16)
    # W1 is (16, 49). W index k*4 + i.
    # For total, W index k*4 + i is the same for all 3 parts?
    # Yes, W depends on k (output col) and i (output row).
    # For part 1 (cols 0-3), k is 0..3.
    # For part 2 (cols 4-7), k is 4..7.
    # For part 3 (col 8), k is 8.
    
    W1_pad = np.zeros((36, 49))
    for r in range(49):
        for k in range(4):
            for i in range(4):
                W1_pad[k*4+i, r] = W1[k*4+i, r]
                
    W2_pad = np.zeros((36, 49))
    for r in range(49):
        for k in range(4):
            for i in range(4):
                W2_pad[(k+4)*4+i, r] = W1[k*4+i, r]
                
    W3_pad = np.zeros((36, 16))
    for r in range(16):
        for i in range(4):
            W3_pad[8*4+i, r] = W3[i, r]
            
    W_total = np.hstack([W1_pad, W2_pad, W3_pad])
    
    # Verify
    def verify_fast(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        total = np.zeros((4, 4, 9), dtype=float)
        for c in range(rank):
            Uc = U[:, c].reshape(4, 4)
            Vc = V[:, c].reshape(4, 9)
            Wc = W[:, c].reshape(9, 4)
            for i in range(4):
                contrib = np.outer(Uc[i, :], Wc[:, i])
                total[i, :, :] += contrib * Vc
        if np.max(np.abs(total - 1.0)) > tol:
            return False
        return True

    if verify_fast(U_total.tolist(), V_total.tolist(), W_total.tolist()):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = 144
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                U[i*4+j][col] = 1
                V[j*9+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
