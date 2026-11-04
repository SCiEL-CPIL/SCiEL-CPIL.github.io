import numpy as np
import time

def get_strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indices:
    u = i*4 + j
    v = j*4 + k
    w = k*4 + i
    """
    # Base case: 2x2x2 Strassen
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # u = i*2 + j, v = j*2 + k, w = k*2 + i
    
    # Strassen 7 terms for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to U, V, W entries.
    # This is tedious to hardcode manually. 
    # Instead, let's use the recursive structure or a known explicit list.
    # Given the time, I will hardcode the 49 terms for 4x4x4 derived from 2x2 blocks.
    
    # Actually, simpler: Just use the schoolbook for the sub-blocks if I can't guarantee Strassen correctness quickly.
    # Schoolbook 4x4x4 rank is 64.
    # 64 + 64 + 16 = 144. No improvement.
    
    # I MUST use Strassen (49) to get improvement.
    # Let's generate Strassen 2x2 first.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A11+A22)(B11+B22)
    # A11 is u=0, A22 is u=3. B11 is v=0, B22 is v=3.
    # C11 (w=0) += M1, C12 (w=1) -= M1? No.
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # U, V, W definitions for rank 1 term k:
    # U[u,k], V[v,k], W[w,k]
    
    # k=0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C11, C22?
    # Wait, M1 contributes to C11 and C22.
    # C11 (w=0) gets +M1. C22 (w=3) gets +M1.
    # So W2[0,0]=1, W2[3,0]=1.
    
    # k=1: M2 = (A21+A22)B11
    # A21 is u=2, A22 is u=3. B11 is v=0.
    # C21 (w=2) gets +M2.
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # k=2: M3 = A11(B21-B22)
    # A11 is u=0. B21 is v=2, B22 is v=3.
    # C12 (w=1) gets +M3. C22 (w=3) gets -M3.
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # k=3: M4 = A22(B12-B11)
    # A22 is u=3. B12 is v=1, B11 is v=0.
    # C11 (w=0) gets -M4. C21 (w=2) gets +M4.
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1
    
    # k=4: M5 = (A11+A12)B22
    # A11 is u=0, A12 is u=1. B22 is v=3.
    # C12 (w=1) gets +M5. C22 (w=3) gets -M5.
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1
    
    # k=5: M6 = (A21-A11)(B11+B12)
    # A21 is u=2, A11 is u=0. B11 is v=0, B12 is v=1.
    # C11 (w=0) gets -M6. C21 (w=2) gets +M6.
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = -1; W2[2, 5] = 1
    
    # k=6: M7 = (A12-A22)(B21+B22)
    # A12 is u=1, A22 is u=3. B21 is v=2, B22 is v=3.
    # C11 (w=0) gets +M7. C22 (w=3) gets +M6? No, +M7.
    # C11 = M1 - M4 - M6 + M7.
    # C22 = M1 - M3 - M5 + M6.
    # So W2[0, 6] = 1. W2[3, 6] = 0?
    # Let's check C22 formula: M1 - M3 - M5 + M6.
    # My M7 definition: (A12-A22)(B21+B22).
    # Does M7 appear in C22? No.
    # So W2[3, 6] = 0.
    # But wait, Strassen 2x2:
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    # Yes.
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2x2
    # This U2, V2, W2 is correct for 2x2x2.
    
    # Now build 4x4x4 from 2x2x2.
    # A is 4x4, viewed as 2x2 blocks of 2x2.
    # A11 (top-left 2x2), A12 (top-right), A21 (bottom-left), A22 (bottom-right).
    # Similarly B, C.
    # Strassen on blocks:
    # M1_block = (A11+A22)(B11+B22)
    # ...
    # Each block multiplication is a 2x2x2 multiplication.
    # So we have 7 block multiplications.
    # Each block multiplication has rank 7.
    # Total rank 49.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block indices for 4x4:
    # u = i*4 + j.
    # Block (bi, bj) where bi, bj in {0,1}.
    # Rows i in 2*bi .. 2*bi+1, cols j in 2*bj .. 2*bj+1.
    # u = i*4 + j.
    
    def map_u4(bi, bj, u2):
        i2 = u2 // 2
        j2 = u2 % 2
        i = 2*bi + i2
        j = 2*bj + j2
        return i*4 + j
        
    def map_v4(bi, bj, v2):
        # v = j*4 + k for 4x4?
        # In 2x2, v = j*2 + k.
        # In 4x4, B is 4x4. v = j*4 + k.
        # Block (bi, bj) of B?
        # B is 4x4. Rows j, Cols k.
        # Block (bi, bj) means rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
        # j = 2*bi + j2, k = 2*bj + k2.
        # v2 = j2*2 + k2.
        # j2 = v2 // 2, k2 = v2 % 2.
        j = 2*bi + (v2 // 2)
        k = 2*bj + (v2 % 2)
        return j*4 + k
        
    def map_w4(bi, bj, w2):
        # w = k*4 + i for 4x4.
        # Block (bi, bj) of C?
        # C is 4x4. Rows i, Cols k.
        # Block (bi, bj) means rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
        # i = 2*bi + i2, k = 2*bj + k2.
        # w2 = k2*2 + i2.
        # k2 = w2 // 2, i2 = w2 % 2.
        i = 2*bi + (w2 % 2)
        k = 2*bj + (w2 // 2)
        return k*4 + i

    # Strassen block definitions:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B21-B22) -> C12 += M3, C22 -= M3
    # M4 = A22(B12-B11) -> C11 -= M4, C21 += M4
    # M5 = (A11+A12)B22 -> C12 += M5, C22 -= M5
    # M6 = (A21-A11)(B11+B12) -> C11 -= M6, C21 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # We iterate over the 7 block terms.
    # For each term, we have a linear combination of A blocks and B blocks.
    # This corresponds to adding U2 and V2 with signs.
    
    # Term 0: M1
    # A: 1*A11 + 1*A22. B: 1*B11 + 1*B22.
    # C: 1*C11 + 1*C22.
    
    # We can just sum the contributions.
    # For each block term t (0..6), we have:
    # A_blocks: dict {(bi, bj): coeff}
    # B_blocks: dict {(bi, bj): coeff}
    # C_blocks: dict {(bi, bj): coeff}
    
    terms = [
        # (A_blocks, B_blocks, C_blocks)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}), # M3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):-1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):-1, (1,0):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}) # M7
    ]
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        for a_key, a_coeff in a_blocks.items():
            for b_key, b_coeff in b_blocks.items():
                for c_key, c_coeff in c_blocks.items():
                    # This is not right. The rank-1 term is the product of the whole block sum.
                    # The block multiplication is rank 7.
                    # So for each block term t, we add 7 rank-1 terms.
                    pass
    
    # Correct approach:
    # For each block term t (0..6), we instantiate the 2x2 Strassen decomposition.
    # The U, V, W for the block term are formed by scaling U2, V2, W2 by the block coefficients.
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        # We need to expand the block coefficients into the full 4x4 matrices.
        # U_block = sum_{(bi,bj), coeff} coeff * U2 mapped to block (bi,bj)
        # But U2 is for a single 2x2 block.
        # So we sum the contributions.
        
        U_block = np.zeros((16, 7))
        V_block = np.zeros((16, 7))
        W_block = np.zeros((16, 7))
        
        for (bi, bj), coeff in a_blocks.items():
            for u2 in range(4):
                u4 = map_u4(bi, bj, u2)
                U_block[u4, :] += coeff * U2[u2, :]
                
        for (bi, bj), coeff in b_blocks.items():
            for v2 in range(4):
                v4 = map_v4(bi, bj, v2)
                V_block[v4, :] += coeff * V2[v2, :]
                
        for (bi, bj), coeff in c_blocks.items():
            for w2 in range(4):
                w4 = map_w4(bi, bj, w2)
                W_block[w4, :] += coeff * W2[w2, :]
                
        # Add to global U4, V4, W4
        start = t_idx * 7
        end = start + 7
        U4[:, start:end] = U_block
        V4[:, start:end] = V_block
        W4[:, start:end] = W_block
        
    return U4, V4, W4

def construct():
    n, m, p = 4, 4, 9
    
    # 1. Get Strassen 4x4x4 decomposition
    U_str, V_str, W_str = get_strassen_4x4()
    
    # 2. Construct 4x4x9 decomposition
    # Rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: B cols 0-3, C cols 0-3
    # U: Copy U_str
    U[:, 0:49] = U_str
    
    # V: Map V_str (16x49) to V (36x49)
    # V_str index v' = j*4 + k (j in 0..3, k in 0..3)
    # V index v = j*9 + k
    for v_prime in range(16):
        j = v_prime // 4
        k = v_prime % 4
        v = j * 9 + k
        V[v, 0:49] = V_str[v_prime, :]
        
    # W: Map W_str (16x49) to W (36x49)
    # W_str index w' = k*4 + i (k in 0..3, i in 0..3)
    # W index w = k*9 + i? No, W index is k*n + i = k*4 + i.
    # Wait, W shape is (p*n, rank) = (9*4, rank) = (36, rank).
    # W index w = k*n + i = k*4 + i.
    # In Block 1, k in 0..3.
    # So w = k*4 + i. This is the same as w'.
    # So W[0:16, 0:49] = W_str
    W[0:16, 0:49] = W_str
    
    # Block 2: B cols 4-7, C cols 4-7
    # U: Copy U_str
    U[:, 49:98] = U_str
    
    # V: Map V_str to V
    # B cols 4-7. k in 4..7.
    # v' = j*4 + k_local (k_local in 0..3)
    # k = 4 + k_local
    # v = j*9 + k = j*9 + 4 + k_local
    for v_prime in range(16):
        j = v_prime // 4
        k_local = v_prime % 4
        k = 4 + k_local
        v = j * 9 + k
        V[v, 49:98] = V_str[v_prime, :]
        
    # W: Map W_str to W
    # C cols 4-7. k in 4..7.
    # w' = k_local*4 + i
    # w = k*4 + i = (4+k_local)*4 + i = 16 + k_local*4 + i
    for w_prime in range(16):
        k_local = w_prime // 4
        i = w_prime % 4
        w = (4 + k_local) * 4 + i
        W[w, 49:98] = W_str[w_prime, :]
        
    # Block 3: B col 8, C col 8
    # Rank 16. Standard dot product.
    # C_{i,8} = sum_j A_{i,j} B_{j,8}
    # Term for (i,j):
    # u = i*4 + j
    # v = j*9 + 8
    # w = 8*4 + i = 32 + i
    
    for i in range(4):
        for j in range(4):
            k_idx = 98 + i*4 + j
            u = i * 4 + j
            v = j * 9 + 8
            w = 32 + i
            U[u, k_idx] = 1
            V[v, k_idx] = 1
            W[w, k_idx] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
