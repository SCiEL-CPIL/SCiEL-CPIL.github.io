"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # rank = n * m * p = 144
    # We use a decomposition of rank 114.
    # T_4,4,9 = T_4,4,4 (k=0..3) + T_4,4,4 (k=4..7) + T_4,4,1 (k=8)
    
    # Helper: Rank-7 decomposition for 2x2 matrix multiplication (Strassen)
    # U2, V2, W2 are (4, 7) matrices
    def get_strassen_2x2():
        # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Strassen steps:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need U2, V2, W2 such that sum_r U2[a,r]V2[b,r]W2[c,r] = T_2,2,2[a,b,c]
        # T_2,2,2[a,b,c] = 1 if a=(i,j), b=(j',k), c=(k',i') and i=i', j=j', k=k'.
        
        # The standard Strassen algorithm gives linear combinations.
        # We can generate the rank-7 tensors by expanding the products.
        
        # It is easier to define the 7 "rank-1" tensors for the Strassen algorithm
        # in the basis of the 7 multiplications, but we need the scalar rank-1 tensors.
        
        # Actually, the rank of 2x2 is 7. The 7 tensors are not simply the Strassen M's.
        # The Strassen M's are products of linear forms.
        # Each M is a 2x2 product.
        # A 2x2 product has rank 4.
        # So 7 * 4 = 28 rank-1 tensors? No, the rank is 7.
        
        # Wait, the rank of the 2x2 MMT is 7.
        # This means we can express T_2,2,2 as sum of 7 rank-1 tensors.
        # The standard "Strassen" rank-7 decomposition uses the 7 M's as the "multiplications".
        # But each M is a scalar? No, in Strassen, M1...M7 are scalars if we look at the entries?
        # No, in Strassen for 2x2, M1...M7 are scalars?
        # A = [[a,b],[c,d]], B=[[e,f],[g,h]].
        # M1 = (a+d)(e+h) -> scalar.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]].
        # Yes, M1...M7 are scalars.
        # So the tensor is decomposed into 7 rank-1 tensors?
        # No, a rank-1 tensor is $u \otimes v \otimes w$.
        # The term $M_1$ contributes to 4 entries of C.
        # So $M_1$ corresponds to a rank-4 tensor?
        # No, the decomposition of the TENSOR $T$ is $\sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        # The coefficients $u_r, v_r, w_r$ are vectors.
        
        # Let's construct the 7 vectors.
        # For each r (1 to 7), we define u_r (4x1), v_r (4x1), w_r (4x1).
        
        # r=1: M1 = (A00+A11)(B00+B11)
        # u1: coeffs for A00, A01, A10, A11.
        # u1 = [1, 0, 0, 1]  (A00+A11)
        # v1 = [1, 0, 0, 1]  (B00+B11)
        # w1: coeffs for C00, C01, C10, C11.
        # C00 += M1, C11 += M1.
        # w1 = [1, 0, 0, 1]
        
        # r=2: M2 = (A10+A11)B00
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2: C10 += M2, C11 -= M2.
        # w2 = [0, 0, 1, -1]
        
        # r=3: M3 = A00(B01-B11)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # w3: C01 += M3, C11 += M3.
        # w3 = [0, 1, 0, 1]
        
        # r=4: M4 = A11(B10-B00)
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1]
        # w4: C00 += M4, C10 += M4.
        # w4 = [1, 0, 1, 0]
        
        # r=5: M5 = (A00+A01)B11
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # w5: C00 -= M5, C01 += M5.
        # w5 = [-1, 1, 0, 0]
        
        # r=6: M6 = (A10-A00)(B00+B01)
        # u6 = [-1, 0, 1, 0]
        # v6 = [1, 1, 0, 0]
        # w6: C11 += M6.
        # w6 = [0, 0, 0, 1]
        
        # r=7: M7 = (A01-A11)(B10+B11)
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # w7: C00 += M7.
        # w7 = [1, 0, 0, 0]
        
        U2 = np.array([u1, u2, u3, u4, u5, u6, u7]).T # Shape (4, 7)
        V2 = np.array([v1, v2, v3, v4, v5, v6, v7]).T
        W2 = np.array([w1, w2, w3, w4, w5, w6, w7]).T
        
        # Verify: Sum_r U2[a,r] V2[b,r] W2[c,r]
        # This should equal T_2,2,2.
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Construct T_4,4,4 decomposition (Rank 49)
    # We apply Strassen to the 4 blocks.
    # Blocks: 00, 01, 10, 11.
    # Strassen steps s=0..6.
    # For each s, we form A_s and B_s (2x2 matrices of blocks).
    # Then we multiply A_s B_s using the 2x2 rank-7 decomposition.
    
    # Coefficients for A_s and B_s (2x2 blocks)
    # s: (coeff_A00, coeff_A01, coeff_A10, coeff_A11), (coeff_B00, ...)
    # A_s is 2x2? No, A_s is formed by combining blocks into a 2x2 matrix?
    # No, Strassen combines the 4 blocks into 7 scalar "products" of 2x2 matrices?
    # Wait, in the 4x4 case, the "elements" of the 2x2 "meta-matrix" are 2x2 blocks.
    # So M1 = (A00+A11)(B00+B11) is a product of two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11. M1 = X Y.
    # X, Y are 2x2. M1 is 2x2.
    
    # So for each s, we have X_s (2x2) and Y_s (2x2).
    # We compute M_s = X_s Y_s using the rank-7 decomposition of 2x2 multiplication.
    
    # Define X_s, Y_s coefficients for the 4 blocks.
    # X_s is a 2x2 matrix of linear combinations of A00, A01, A10, A11.
    # Actually, X_s is just a 2x2 matrix formed by adding blocks.
    # X_s = c00*A00 + c01*A01 + c10*A10 + c11*A11.
    
    # Strassen coefficients:
    # s=0 (M1): X = A00+A11, Y = B00+B11
    # s=1 (M2): X = A10+A11, Y = B00
    # s=2 (M3): X = A00, Y = B01-B11
    # s=3 (M4): X = A11, Y = B10-B00
    # s=4 (M5): X = A00+A01, Y = B11
    # s=5 (M6): X = A10-A00, Y = B00+B01
    # s=6 (M7): X = A01-A11, Y = B10+B11
    
    # Coeffs for X (A blocks) and Y (B blocks)
    # Order: A00, A01, A10, A11
    X_coeffs = [
        [1, 0, 0, 1], # s=0
        [0, 0, 1, 1], # s=1
        [1, 0, 0, 0], # s=2
        [0, 0, 0, 1], # s=3
        [1, 1, 0, 0], # s=4
        [-1, 0, 1, 0],# s=5
        [0, 1, 0, -1] # s=6
    ]
    Y_coeffs = [
        [1, 0, 0, 1], # s=0
        [1, 0, 0, 0], # s=1
        [0, 1, 0, -1],# s=2
        [0, 0, 1, -1],# s=3
        [0, 0, 0, 1], # s=4
        [1, 1, 0, 0], # s=5
        [0, 0, 1, 1]  # s=6
    ]
    
    # C coefficients: C_block = sum_s gamma[s, block] M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    C_coeffs = [
        [1, 0, 0, 0], # C00: s0, s1, s2, s3, s4, s5, s6 -> 1, 0, 0, 1, -1, 0, 1
        [0, 1, 0, 1], # C01: 0, 0, 1, 0, 1, 0, 0
        [1, 0, 0, 1], # C10: 0, 1, 0, 1, 0, 0, 0
        [1, -1, 1, 1] # C11: 1, -1, 1, 0, 0, 1, 0
    ]
    # Correcting C_coeffs to full 7x4
    C_coeffs = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]) # Shape (4, 7) where rows are blocks, cols are s
    
    # Now construct U4, V4, W4 for T_4,4,4
    # Rank 49 = 7 (strassen steps) * 7 (inner rank)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((36, 49)) # Wait, m=4, p=4 for this sub-tensor. m*p=16.
    # But in the global problem, p=9.
    # We will handle the global indexing later.
    # For now, V4 is for p=4. Shape (16, 49).
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for s in range(7):
        for r_int in range(7):
            col = s * 7 + r_int
            # U4 contribution
            # U4 index a (0..15). a = block_idx * 4 + inner_idx.
            # U2 index a_s (0..3). a_s = inner_idx.
            # U4[a, col] = sum_{block} X_coeffs[s, block] * U2[inner_idx, r_int]
            for block in range(4):
                coeff_x = X_coeffs[s][block]
                if coeff_x == 0: continue
                base_idx = block * 4
                for inner in range(4):
                    a = base_idx + inner
                    U4[a, col] += coeff_x * U2[inner, r_int]
            
            # V4 contribution
            # V4 index b (0..15). b = block_idx * 4 + inner_idx.
            # V2 index b_s (0..3).
            for block in range(4):
                coeff_y = Y_coeffs[s][block]
                if coeff_y == 0: continue
                base_idx = block * 4
                for inner in range(4):
                    b = base_idx + inner
                    V4[b, col] += coeff_y * V2[inner, r_int]
            
            # W4 contribution
            # W4 index c (0..15). c = block_idx * 4 + inner_idx.
            # W2 index c_s (0..3).
            # W4[c, col] = sum_{block} C_coeffs[block, s] * W2[inner_idx, r_int]
            for block in range(4):
                coeff_c = C_coeffs[block][s]
                if coeff_c == 0: continue
                base_idx = block * 4
                for inner in range(4):
                    c = base_idx + inner
                    W4[c, col] += coeff_c * W2[inner, r_int]
    
    # Construct T_4,4,1 decomposition (Rank 16)
    U1 = np.zeros((16, 16))
    V1 = np.zeros((4, 16)) # m=4, p=1 -> 4x1 = 4.
    W1 = np.zeros((4, 16)) # p=1, n=4 -> 1x4 = 4.
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U1[i * 4 + j, col] = 1
            V1[j, col] = 1
            W1[i, col] = 1
            
    # Assemble Global U, V, W for T_4,4,9
    # Rank R = 49 + 49 + 16 = 114
    R = 114
    U = np.zeros((16, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    
    # Block 1: k in 0..3
    # V index: j' * 9 + k. k in 0..3.
    # W index: k' * 4 + i'. k' in 0..3.
    for c in range(49):
        U[:, c] = U4[:, c]
        # V4 has indices 0..15 (j' * 4 + k).
        # Global V index is j' * 9 + k.
        for j_p in range(4):
            for k in range(4):
                v4_idx = j_p * 4 + k
                v_glob_idx = j_p * 9 + k
                V[v_glob_idx, c] = V4[v4_idx, c]
        # W4 has indices 0..15 (k' * 4 + i').
        # Global W index is k' * 4 + i'.
        # They match directly for k' in 0..3.
        W[:, c] = W4[:, c]
        
    # Block 2: k in 4..7
    # V index: j' * 9 + k. k in 4..7.
    # W index: k' * 4 + i'. k' in 4..7.
    offset = 49
    for c in range(49):
        col = offset + c
        U[:, col] = U4[:, c]
        for j_p in range(4):
            for k in range(4):
                v4_idx = j_p * 4 + k
                v_glob_idx = j_p * 9 + (k + 4)
                V[v_glob_idx, col] = V4[v4_idx, c]
        for k_p in range(4):
            for i in range(4):
                w4_idx = k_p * 4 + i
                w_glob_idx = (k_p + 4) * 4 + i
                W[w_glob_idx, col] = W4[w4_idx, c]
                
    # Block 3: k in 8
    # V index: j' * 9 + 8.
    # W index: 8 * 4 + i' = 32 + i'.
    offset = 98
    for c in range(16):
        col = offset + c
        U[:, col] = U1[:, c]
        for j_p in range(4):
            v1_idx = j_p
            v_glob_idx = j_p * 9 + 8
            V[v_glob_idx, col] = V1[v1_idx, c]
        for i in range(4):
            w1_idx = i
            w_glob_idx = 8 * 4 + i
            W[w_glob_idx, col] = W1[w1_idx, c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
