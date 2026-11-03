import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    # Build the true tensor entries
    tensor_entries = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                tensor_entries.append((i, j, k))
    
    entry_set = set(tensor_entries)

    def verify_fast(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        T_true = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for (i, j, k) in tensor_entries:
            T_true[i, j, k] = 1.0
        return np.allclose(T, T_true, atol=1e-9, rtol=0)

    def make_strassen_4x4x4():
        """
        Construct rank-49 Strassen decomposition for 4x4x4.
        Returns U, V, W of size 16x49.
        """
        # Strassen for 2x2x2 has rank 7.
        # We apply it to the 2x2 blocks of the 4x4 matrices.
        # This results in 7 products of 2x2 matrices.
        # Each 2x2 product is computed with Strassen (rank 7).
        # Total rank 49.
        
        # Indices for 4x4 matrices:
        # A: a11 a12 a13 a14
        #    a21 a22 a23 a24
        #    a31 a32 a33 a34
        #    a41 a42 a43 a44
        # Index = row * 4 + col
        
        # Strassen 2x2 blocks:
        # A11 = [[a11, a12], [a21, a22]]
        # A12 = [[a13, a14], [a23, a24]]
        # A21 = [[a31, a32], [a41, a42]]
        # A22 = [[a33, a34], [a43, a44]]
        
        # The 7 Strassen products for the 4x4 level:
        # M1 = (A11 + A22) (B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # Each Mk is a 2x2 matrix.
        # We compute each Mk using Strassen 2x2x2.
        
        # Strassen 2x2x2 matrices (rank 7):
        # U (4x7), V (4x7), W (4x7)
        # Indices 0,1,2,3 for 2x2 matrix.
        
        def strassen_2x2():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1 = (A00+A11)(B00+B11)
            U[0, 0] = 1; U[1, 0] = 1
            V[0, 0] = 1; V[1, 0] = 1
            W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
            
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # U (A)
            # M1: A00+A11 -> U[0,0]=1, U[3,0]=1
            # M2: A10+A11 -> U[2,1]=1, U[3,1]=1
            # M3: A00 -> U[0,2]=1
            # M4: A11 -> U[3,3]=1
            # M5: A00+A01 -> U[0,4]=1, U[1,4]=1
            # M6: A10-A00 -> U[2,5]=1, U[0,5]=-1
            # M7: A01-A11 -> U[1,6]=1, U[3,6]=-1
            
            U[0, 0] = 1; U[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            U[0, 2] = 1
            U[3, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            U[1, 6] = 1; U[3, 6] = -1
            
            # V (B)
            # M1: B00+B11 -> V[0,0]=1, V[3,0]=1
            # M2: B00 -> V[0,1]=1
            # M3: B01-B11 -> V[1,2]=1, V[3,2]=-1
            # M4: B10-B00 -> V[2,3]=1, V[0,3]=-1
            # M5: B11 -> V[3,4]=1
            # M6: B00+B01 -> V[0,5]=1, V[1,5]=1
            # M7: B10+B11 -> V[2,6]=1, V[3,6]=1
            
            V[0, 0] = 1; V[3, 0] = 1
            V[0, 1] = 1
            V[1, 2] = 1; V[3, 2] = -1
            V[2, 3] = 1; V[0, 3] = -1
            V[3, 4] = 1
            V[0, 5] = 1; V[1, 5] = 1
            V[2, 6] = 1; V[3, 6] = 1
            
            # W (C)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
            W[1, 2] = 1; W[1, 4] = 1
            W[2, 1] = 1; W[2, 3] = 1
            W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
            
            return U, V, W
        
        U_2x2, V_2x2, W_2x2 = strassen_2x2()
        
        # Now construct 4x4x4 Strassen
        U_4x4 = np.zeros((16, 49))
        V_4x4 = np.zeros((16, 49))
        W_4x4 = np.zeros((16, 49))
        
        # Block indices for 4x4:
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mk, we use the 2x2 Strassen matrices.
        # The 2x2 matrices for A in Mk are linear combinations of the blocks.
        
        # M1: A_part = A11+A22, B_part = B11+B22
        # A11 indices: 0,1,4,5 -> 2x2 indices 0,1,2,3
        # A22 indices: 10,11,14,15 -> 2x2 indices 0,1,2,3
        # So U for M1: U_2x2 with indices mapped.
        
        def map_2x2_to_4x4(U_2x2, block_indices):
            # block_indices: list of 4 indices in the 4x4 matrix corresponding to the 2x2 matrix
            U_new = np.zeros((16, 7))
            for i in range(4):
                for r in range(7):
                    U_new[block_indices[i], r] = U_2x2[i, r]
            return U_new
        
        # A blocks
        A11_idx = [0, 1, 4, 5]
        A12_idx = [2, 3, 6, 7]
        A21_idx = [8, 9, 12, 13]
        A22_idx = [10, 11, 14, 15]
        
        # B blocks
        B11_idx = [0, 1, 4, 5]
        B12_idx = [2, 3, 6, 7]
        B21_idx = [8, 9, 12, 13]
        B22_idx = [10, 11, 14, 15]
        
        # C blocks
        C11_idx = [0, 1, 4, 5]
        C12_idx = [2, 3, 6, 7]
        C21_idx = [8, 9, 12, 13]
        C22_idx = [10, 11, 14, 15]
        
        # M1: (A11+A22)(B11+B22)
        U_M1 = map_2x2_to_4x4(U_2x2, A11_idx) + map_2x2_to_4x4(U_2x2, A22_idx)
        V_M1 = map_2x2_to_4x4(V_2x2, B11_idx) + map_2x2_to_4x4(V_2x2, B22_idx)
        W_M1 = map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C12_idx) + map_2x2_to_4x4(W_2x2, C21_idx) + map_2x2_to_4x4(W_2x2, C22_idx)
        # Wait, W for 2x2 Strassen computes the 2x2 result.
        # M1 is a 2x2 matrix. Its result contributes to C11.
        # So W_M1 should be map_2x2_to_4x4(W_2x2, C11_idx).
        
        W_M1 = map_2x2_to_4x4(W_2x2, C11_idx)
        
        # M2: (A21+A22)B11
        U_M2 = map_2x2_to_4x4(U_2x2, A21_idx) + map_2x2_to_4x4(U_2x2, A22_idx)
        V_M2 = map_2x2_to_4x4(V_2x2, B11_idx)
        W_M2 = map_2x2_to_4x4(W_2x2, C21_idx) # M2 contributes to C21
        
        # M3: A11(B12-B22)
        U_M3 = map_2x2_to_4x4(U_2x2, A11_idx)
        V_M3 = map_2x2_to_4x4(V_2x2, B12_idx) - map_2x2_to_4x4(V_2x2, B22_idx)
        W_M3 = map_2x2_to_4x4(W_2x2, C12_idx) # M3 contributes to C12
        
        # M4: A22(B21-B11)
        U_M4 = map_2x2_to_4x4(U_2x2, A22_idx)
        V_M4 = map_2x2_to_4x4(V_2x2, B21_idx) - map_2x2_to_4x4(V_2x2, B11_idx)
        W_M4 = map_2x2_to_4x4(W_2x2, C22_idx) # M4 contributes to C22? No, C22 = M1 - M2 + M3 + M6.
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # C11 (top-left) gets M1, M4, M5, M7
        # C12 (top-right) gets M3, M5
        # C21 (bottom-left) gets M2, M4
        # C22 (bottom-right) gets M1, M2, M3, M6
        
        # Let's re-verify the Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # W_M1 contributes to C11 (C00) and C22 (C11).
        # W_M1 = map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C22_idx)
        
        # W_M2 contributes to C21 (C10) and C22 (C11) with -1.
        # W_M2 = map_2x2_to_4x4(W_2x2, C21_idx) - map_2x2_to_4x4(W_2x2, C22_idx)
        
        # W_M3 contributes to C12 (C01) and C22 (C11).
        # W_M3 = map_2x2_to_4x4(W_2x2, C12_idx) + map_2x2_to_4x4(W_2x2, C22_idx)
        
        # W_M4 contributes to C11 (C00) and C21 (C10).
        # W_M4 = map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C21_idx)
        
        # W_M5 contributes to C11 (C00) with -1 and C12 (C01).
        # W_M5 = -map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C12_idx)
        
        # W_M6 contributes to C22 (C11).
        # W_M6 = map_2x2_to_4x4(W_2x2, C22_idx)
        
        # W_M7 contributes to C11 (C00).
        # W_M7 = map_2x2_to_4x4(W_2x2, C11_idx)
        
        # Let's re-calculate W_M1..W_M7 with correct contributions.
        
        W_M1 = map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C22_idx)
        W_M2 = map_2x2_to_4x4(W_2x2, C21_idx) - map_2x2_to_4x4(W_2x2, C22_idx)
        W_M3 = map_2x2_to_4x4(W_2x2, C12_idx) + map_2x2_to_4x4(W_2x2, C22_idx)
        W_M4 = map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C21_idx)
        W_M5 = -map_2x2_to_4x4(W_2x2, C11_idx) + map_2x2_to_4x4(W_2x2, C12_idx)
        W_M6 = map_2x2_to_4x4(W_2x2, C22_idx)
        W_M7 = map_2x2_to_4x4(W_2x2, C11_idx)
        
        # Now assemble U_4x4, V_4x4, W_4x4
        # 7 blocks of 7 columns each.
        
        for k in range(7):
            offset = k * 7
            if k == 0:
                U_4x4[:, offset:offset+7] = U_M1
                V_4x4[:, offset:offset+7] = V_M1
                W_4x4[:, offset:offset+7] = W_M1
            elif k == 1:
                U_4x4[:, offset:offset+7] = U_M2
                V_4x4[:, offset:offset+7] = V_M2
                W_4x4[:, offset:offset+7] = W_M2
            elif k == 2:
                U_4x4[:, offset:offset+7] = U_M3
                V_4x4[:, offset:offset+7] = V_M3
                W_4x4[:, offset:offset+7] = W_M3
            elif k == 3:
                U_4x4[:, offset:offset+7] = U_M4
                V_4x4[:, offset:offset+7] = V_M4
                W_4x4[:, offset:offset+7] = W_M4
            elif k == 4:
                U_4x4[:, offset:offset+7] = U_M5
                V_4x4[:, offset:offset+7] = V_M5
                W_4x4[:, offset:offset+7] = W_M5
            elif k == 5:
                U_4x4[:, offset:offset+7] = U_M6
                V_4x4[:, offset:offset+7] = V_M6
                W_4x4[:, offset:offset+7] = W_M6
            elif k == 6:
                U_4x4[:, offset:offset+7] = U_M7
                V_4x4[:, offset:offset+7] = V_M7
                W_4x4[:, offset:offset+7] = W_M7
        
        return U_4x4, V_4x4, W_4x4

    U_4x4, V_4x4, W_4x4 = make_strassen_4x4x4()
    
    # Construct 4x4x8 decomposition using two 4x4x4 decompositions
    # Rank 98
    U_4x4x8 = np.zeros((16, 98))
    V_4x4x8 = np.zeros((32, 98))
    W_4x4x8 = np.zeros((32, 98))
    
    # First 4 columns of B and C
    U_4x4x8[:, 0:49] = U_4x4
    V_4x4x8[0:16, 0:49] = V_4x4 # B11, B12 (cols 0-3)
    # V indices for B: b*p + c.
    # For first 4 cols of B (c=0..3), b=0..3.
    # j = b*8 + c.
    # For b=0, c=0..3 -> 0..3
    # For b=1, c=0..3 -> 8..11
    # For b=2, c=0..3 -> 16..19
    # For b=3, c=0..3 -> 24..27
    # So V_4x4 indices 0..15 need to be mapped to 0..31.
    
    # V_4x4 is 16x49.
    # V_4x4[i] corresponds to B[i//4, i%4].
    # In 4x4x8, B is 4x8.
    # B[b, c] for c=0..3 is at index b*8+c.
    # So V_4x4[b*4+c] -> V_4x4x8[b*8+c].
    
    for b in range(4):
        for c in range(4):
            V_4x4x8[b*8+c, 0:49] = V_4x4[b*4+c, 0:49]
    
    # W_4x4 is 16x49.
    # W_4x4[k] corresponds to C[k//4, k%4]?
    # W index = c*n + a.
    # For 4x4x4, n=4, p=4.
    # W index = c*4 + a.
    # For 4x4x8, n=4, p=8.
    # W index = c*4 + a.
    # For c=0..3, the indices are the same.
    # So W_4x4x8[0:16, 0:49] = W_4x4.
    
    W_4x4x8[0:16, 0:49] = W_4x4
    
    # Last 4 columns of B and C
    U_4x4x8[:, 49:98] = U_4x4
    # V for c=4..7
    for b in range(4):
        for c in range(4, 8):
            V_4x4x8[b*8+c, 49:98] = V_4x4[b*4+(c-4), 49:98]
            
    # W for c=4..7
    # W index = c*4 + a.
    # For c=4, indices 16..19.
    # For c=5, indices 20..23.
    # For c=6, indices 24..27.
    # For c=7, indices 28..31.
    
    for c in range(4, 8):
        W_4x4x8[c*4:(c+1)*4, 49:98] = W_4x4[(c-4)*4:(c-3)*4, 49:98]
        
    # Verify
    if verify_fast(U_4x4x8.tolist(), V_4x4x8.tolist(), W_4x4x8.tolist()):
        return {"U": U_4x4x8.tolist(), "V": V_4x4x8.tolist(), "W": W_4x4x8.tolist()}
    
    # Fallback to schoolbook
    U_school = [[0] * 128 for _ in range(16)]
    V_school = [[0] * 128 for _ in range(32)]
    W_school = [[0] * 128 for _ in range(32)]
    col = 0
    for (i, j, k) in tensor_entries:
        U_school[i][col] = 1
        V_school[j][col] = 1
        W_school[k][col] = 1
        col += 1
        
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
