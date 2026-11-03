import time
import sys
import random

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    # The current seed uses a trivial rank-1 decomposition with rank = n*m*p = 112.
    # We can improve this by using the Strassen algorithm for the 4x4x4 sub-problem.
    # The 4x4x4 matrix multiplication tensor can be decomposed with rank 49 (Strassen).
    # The remaining part is a 4x4x3 matrix multiplication, which we can decompose trivially with rank 4*4*3 = 48.
    # Total rank = 49 + 48 = 97.
    # 97 < 112, so this is an improvement.
    
    # We need to construct the U, V, W matrices for the 4x4x4 Strassen decomposition.
    # Then we need to construct the U, V, W matrices for the 4x4x3 trivial decomposition.
    # Finally, we concatenate them.
    
    # Helper function to get the Strassen 2x2x2 decomposition
    def get_strassen_2x2_2x2_UVW():
        # U is 4x7, V is 4x7, W is 4x7
        # Indices: U[i*2+j, t], V[i*2+j, t], W[i*2+j, t]
        # i, j in 0..1, t in 0..6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen formulas:
        # M0 = (A00 + A11)(B00 + B11)
        # M1 = (A10 + A11)B00
        # M2 = A00(B01 - B11)
        # M3 = A11(B10 - B00)
        # M4 = (A00 + A01)B11
        # M5 = (A10 - A00)(B00 + B01)
        # M6 = (A01 - A11)(B10 + B11)
        
        # C00 = M0 + M3 - M4 - M6
        # C01 = M2 + M4
        # C10 = M1 + M5
        # C11 = M0 - M1 + M2 + M5
        
        # t=0: M0
        U[0][0] = 1; U[3][0] = 1  # A00 + A11
        V[0][0] = 1; V[3][0] = 1  # B00 + B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # t=1: M1
        U[2][1] = 1; U[3][1] = 1  # A10 + A11
        V[0][1] = 1               # B00
        W[2][1] = 1; W[3][1] = -1 # C10, -C11
        
        # t=2: M2
        U[0][2] = 1               # A00
        V[1][2] = 1; V[3][2] = -1 # B01 - B11
        W[1][2] = 1; W[3][2] = 1  # C01, C11
        
        # t=3: M3
        U[3][3] = 1               # A11
        V[2][3] = 1; V[0][3] = -1 # B10 - B00
        W[0][3] = 1               # C00
        
        # t=4: M4
        U[0][4] = 1; U[1][4] = 1  # A00 + A01
        V[3][4] = 1               # B11
        W[0][4] = -1; W[1][4] = 1 # -C00, C01
        
        # t=5: M5
        U[2][5] = 1; U[0][5] = -1 # A10 - A00
        V[0][5] = 1; V[1][5] = 1  # B00 + B01
        W[2][5] = 1; W[3][5] = 1  # C10, C11
        
        # t=6: M6
        U[1][6] = 1; U[3][6] = -1 # A01 - A11
        V[2][6] = 1; V[3][6] = 1  # B10 + B11
        W[0][6] = -1              # -C00
        
        return U, V, W
    
    # Build 4x4x4 Strassen decomposition
    # We partition the 4x4 matrices into 2x2 blocks.
    # A is 4x4, B is 4x4, C is 4x4.
    # A[i, j], B[i, j], C[i, j] for i, j in 0..3.
    # The 2x2 Strassen decomposition works on 2x2 blocks.
    # Let A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4].
    # Similarly for B and C.
    
    # The 2x2 Strassen decomposition has U2, V2, W2 of shape 4x7.
    # The 4x4 Strassen decomposition will have U4, V4, W4 of shape 16x49.
    # For each of the 7 Strassen terms, we have a 2x2 multiplication.
    # So we need to map the 2x2 indices to the 4x4 indices.
    
    # U4[i, j, t] where t = 7*block_idx + local_t
    # block_idx in 0..3 (corresponding to the 4 blocks of the 4x4 matrix)
    # local_t in 0..6 (the 7 Strassen terms)
    
    # The 2x2 Strassen decomposition is for a single 2x2 block multiplication.
    # But the 4x4 matrix multiplication involves 4 block multiplications? No.
    # Wait, the 4x4 matrix multiplication is a single tensor.
    # The Strassen algorithm for 4x4x4 uses 7 multiplications of 2x2 matrices.
    # But the 2x2 matrices themselves are composed of 4x4 entries.
    # This is recursive.
    
    # Let's use the direct approach: the 4x4x4 tensor is the Kronecker product of the 2x2x2 tensor with itself?
    # No, that's not exactly right.
    # Actually, the 4x4x4 multiplication tensor can be decomposed using the 2x2x2 Strassen decomposition by treating the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A = [[A00, A01], [A10, A11]] where each Aij is a 2x2 matrix.
    # Similarly for B and C.
    # Then C = [[C00, C01], [C10, C11]] where each Cij is a 2x2 matrix.
    # The 2x2 Strassen algorithm gives:
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    # where M0 = (A00 + A11)(B00 + B11), etc.
    # Each M is a 2x2 matrix product.
    # So we need to decompose each 2x2 matrix product.
    # But the 2x2 matrix product itself is a 2x2x2 tensor.
    # So we can use the 2x2x2 Strassen decomposition for each of the 7 terms.
    # This gives a total of 7 * 7 = 49 rank-1 terms.
    
    # Let's construct this.
    U2, V2, W2 = get_strassen_2x2_2x2_UVW()
    
    # The 4x4x4 decomposition will have U4, V4, W4 of shape 16x49.
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 Strassen terms (t2 in 0..6), we have a 2x2 multiplication.
    # The 2x2 multiplication is decomposed using U2, V2, W2.
    # For each local term (t2_local in 0..6), we have a rank-1 term.
    # The global term index is t = 7*t2 + t2_local.
    
    # The 2x2 blocks of A are:
    # A00 = A[0:2, 0:2] -> indices (0,0), (0,1), (1,0), (1,1) -> U4 indices 0,1,4,5
    # A01 = A[0:2, 2:4] -> indices (0,2), (0,3), (1,2), (1,3) -> U4 indices 2,3,6,7
    # A10 = A[2:4, 0:2] -> indices (2,0), (2,1), (3,0), (3,1) -> U4 indices 8,9,12,13
    # A11 = A[2:4, 2:4] -> indices (2,2), (2,3), (3,2), (3,3) -> U4 indices 10,11,14,15
    
    # Similarly for B and C.
    
    # The 2x2 Strassen algorithm uses:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # For each M, we need to form the 2x2 matrices X and Y.
    # For example, for M0:
    # X = A00 + A11, Y = B00 + B11.
    # The entries of X are linear combinations of A entries.
    # X[i, j] = A00[i, j] + A11[i, j].
    # So the U4 coefficients for the A00 part are the same as U2 coefficients.
    # And the U4 coefficients for the A11 part are the same as U2 coefficients.
    
    # Let's define the block mappings.
    # block_map[block_idx] = list of 4 U4 indices corresponding to the 2x2 block.
    # block_idx 0: A00 -> [0, 1, 4, 5]
    # block_idx 1: A01 -> [2, 3, 6, 7]
    # block_idx 2: A10 -> [8, 9, 12, 13]
    # block_idx 3: A11 -> [10, 11, 14, 15]
    
    block_map = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # For each Strassen term t2 in 0..6, we need to identify which blocks are involved and with what coefficients.
    # For example, for t2=0 (M0):
    # X = A00 + A11, so blocks 0 and 3 with coefficient 1.
    # Y = B00 + B11, so blocks 0 and 3 with coefficient 1.
    # For t2=1 (M1):
    # X = A10 + A11, so blocks 2 and 3 with coefficient 1.
    # Y = B00, so block 0 with coefficient 1.
    
    # Let's define the coefficients for each Strassen term.
    # coeff[t2] = (X_blocks, Y_blocks) where X_blocks is a list of (block_idx, coeff) and Y_blocks is a list of (block_idx, coeff).
    
    coeff = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M0
        ([(2, 1), (3, 1)], [(0, 1)]),           # M1
        ([(0, 1)], [(1, 1), (3, -1)]),          # M2
        ([(3, 1)], [(2, 1), (0, -1)]),          # M3
        ([(0, 1), (1, 1)], [(3, 1)]),           # M4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # M5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])   # M6
    ]
    
    # The W coefficients for the 4x4 C matrix.
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    
    # So for each Strassen term t2, the W coefficients for the 4 blocks are:
    # t2=0 (M0): C00 += 1, C11 += 1
    # t2=1 (M1): C10 += 1, C11 -= 1
    # t2=2 (M2): C01 += 1, C11 += 1
    # t2=3 (M3): C00 += 1
    # t2=4 (M4): C00 -= 1, C01 += 1
    # t2=5 (M5): C10 += 1, C11 += 1
    # t2=6 (M6): C00 -= 1
    
    w_coeff = [
        [(0, 1), (3, 1)],  # M0
        [(2, 1), (3, -1)], # M1
        [(1, 1), (3, 1)],  # M2
        [(0, 1)],          # M3
        [(0, -1), (1, 1)], # M4
        [(2, 1), (3, 1)],  # M5
        [(0, -1)]          # M6
    ]
    
    # Now we can fill U4, V4, W4.
    for t2 in range(7):
        X_blocks, Y_blocks = coeff[t2]
        for t2_local in range(7):
            t_global = 7 * t2 + t2_local
            # For each (i2, j2) in 0..1, we have a U2[i2*2+j2, t2_local] coefficient.
            # This coefficient applies to the corresponding A block entries.
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][t2_local]
                    v2_val = V2[i2*2+j2][t2_local]
                    w2_val = W2[i2*2+j2][t2_local]
                    
                    # For U4, we need to add u2_val * coeff to U4[block_map[block_idx][i2*2+j2], t_global]
                    for block_idx, c_val in X_blocks:
                        u4_idx = block_map[block_idx][i2*2+j2]
                        U4[u4_idx][t_global] += u2_val * c_val
                    
                    # For V4, we need to add v2_val * coeff to V4[block_map[block_idx][i2*2+j2], t_global]
                    for block_idx, c_val in Y_blocks:
                        v4_idx = block_map[block_idx][i2*2+j2]
                        V4[v4_idx][t_global] += v2_val * c_val
                    
                    # For W4, we need to add w2_val * coeff to W4[block_map[block_idx][i2*2+j2], t_global]
                    for block_idx, c_val in w_coeff[t2]:
                        w4_idx = block_map[block_idx][i2*2+j2]
                        W4[w4_idx][t_global] += w2_val * c_val
    
    # Now we have the 4x4x4 Strassen decomposition with rank 49.
    # We need to combine it with the 4x4x3 trivial decomposition.
    # The 4x4x3 trivial decomposition has rank 48.
    # We need to construct U3, V3, W3 for the 4x4x3 case.
    
    # The 4x4x3 multiplication tensor:
    # A is 4x4, B is 4x3, C is 4x3.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The trivial decomposition has rank 4*4*3 = 48.
    # For each (i, j, k) in 0..3, 0..3, 0..2, we have a rank-1 term.
    # U[i*4+j, t] = 1, V[j*3+k, t] = 1, W[k*4+i, t] = 1.
    
    U3 = [[0]*48 for _ in range(16)]
    V3 = [[0]*48 for _ in range(12)]
    W3 = [[0]*48 for _ in range(12)]
    
    t3 = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u3_idx = i*4+j
                v3_idx = j*3+k
                w3_idx = k*4+i
                U3[u3_idx][t3] = 1
                V3[v3_idx][t3] = 1
                W3[w3_idx][t3] = 1
                t3 += 1
    
    # Now we need to combine the 4x4x4 and 4x4x3 decompositions.
    # The full 4x4x7 multiplication tensor is the direct sum of the 4x4x4 and 4x4x3 tensors.
    # So we can concatenate the U, V, W matrices.
    
    # U is 16x(49+48) = 16x97
    # V is (16+12)x(49+48) = 28x97? No.
    # Wait, the shapes are:
    # U: (n*m, rank) = (16, 97)
    # V: (m*p, rank) = (4*7, 97) = (28, 97)
    # W: (p*n, rank) = (7*4, 97) = (28, 97)
    
    # The 4x4x4 part:
    # U4: 16x49
    # V4: 16x49 (but V should be 28x49? No, V4 is for the 4x4 B matrix, which is 16 entries. But in the 4x4x7 case, B is 4x7, so V should be 28x97.)
    # Ah, I see. The 4x4x4 part uses the first 4 columns of B.
    # The 4x4x3 part uses the last 3 columns of B.
    # So V4 should be 28x49, with the first 16 rows corresponding to the 4x4 part and the last 12 rows zero.
    # Similarly, W4 should be 28x49, with the first 16 rows corresponding to the 4x4 part and the last 12 rows zero.
    # And U4 is 16x49.
    
    # Let's adjust.
    V4_full = [[0]*49 for _ in range(28)]
    W4_full = [[0]*49 for _ in range(28)]
    
    for i in range(16):
        for t in range(49):
            V4_full[i][t] = V4[i][t]
            W4_full[i][t] = W4[i][t]
    
    # The 4x4x3 part:
    # U3: 16x48
    # V3: 12x48 (but V should be 28x48, with the first 16 rows zero and the last 12 rows corresponding to the 4x3 part)
    # W3: 12x48 (but W should be 28x48, with the first 16 rows zero and the last 12 rows corresponding to the 4x3 part)
    
    U3_full = [[0]*48 for _ in range(16)]
    V3_full = [[0]*48 for _ in range(28)]
    W3_full = [[0]*48 for _ in range(28)]
    
    for i in range(16):
        for t in range(48):
            U3_full[i][t] = U3[i][t]
    
    for i in range(12):
        for t in range(48):
            V3_full[16+i][t] = V3[i][t]
            W3_full[16+i][t] = W3[i][t]
    
    # Now we can concatenate.
    rank = 49 + 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
        for t in range(48):
            U[i][49+t] = U3_full[i][t]
    
    for i in range(28):
        for t in range(49):
            V[i][t] = V4_full[i][t]
        for t in range(48):
            V[i][49+t] = V3_full[i][t]
    
    for i in range(28):
        for t in range(49):
            W[i][t] = W4_full[i][t]
        for t in range(48):
            W[i][49+t] = W3_full[i][t]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
