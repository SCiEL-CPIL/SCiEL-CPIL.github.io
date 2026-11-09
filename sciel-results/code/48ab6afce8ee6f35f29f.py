import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 49 + 32 + 24 = 105
    rank = 105
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 B1a (4x4 x 4) using Strassen
    # A1 is A[0:4, 0:4]
    # B1a is B[0:4, 0:4]
    # Indices in U: i in 0..3, j in 0..3 -> i*m + j
    # Indices in V: j in 0..3, k in 0..3 -> j*p + k
    # Indices in W: k in 0..3, i in 0..3 -> k*n + i
    
    # Strassen 2x2x2 decomposition for 4x4x4
    # We need to map the 7 Strassen terms for each of the 7 block products?
    # No, Strassen on 4x4x4 uses 7 products of 2x2x2 blocks.
    # Each 2x2x2 product has rank 7.
    # So we iterate over the 7 Strassen "blocks" and for each, add 7 terms.
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to decompose each M_k (which is a 2x2 matrix product) into 7 rank-1 terms.
    # The rank-1 terms for 2x2x2 are:
    # T1: (1,0,0) x (0,1,0) x (0,0,1) ? No.
    # Standard Strassen basis for 2x2x2:
    # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [0,1,0,1] ... this is getting complex to hardcode.
    
    # Alternative: Use the known rank-7 decomposition for 2x2x2.
    # The 7 terms are:
    # 1. (1,0) (0,1) -> (0,0) ? No.
    # Let's use the explicit Strassen identity terms.
    # For 2x2 matrices A, B, C:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # We will add 7 terms for each of the 7 Strassen block products.
    # Total 49 terms.
    
    # Helper to add a 2x2x2 Strassen decomposition
    # offset_i, offset_j for A (2x2 block)
    # offset_j, offset_k for B (2x2 block)
    # offset_k, offset_i for C (2x2 block)
    
    def add_strassen_2x2x2(offset_i, offset_j, offset_k, start_col):
        # The 7 terms of the Strassen decomposition of the 2x2x2 tensor
        # These are the rank-1 components.
        # Term 1: (A11+A22)(B11+B22)
        # This corresponds to specific combinations of indices.
        # Actually, it's easier to just list the 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
        
        # The 7 rank-1 tensors for 2x2x2 are:
        # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [0,1,0,1] ? No.
        
        # Let's use the standard basis:
        # T1 = (1,0,0,0) x (0,1,0,0) x (0,0,1,0) -> C11 = A12 B21 ? No.
        
        # Correct Strassen terms for 2x2x2:
        # The multiplication tensor T_2 has rank 7.
        # The 7 vectors are:
        # 1. u=[1,0,0,1], v=[1,0,0,1], w=[0,1,0,1]
        # 2. u=[0,1,1,0], v=[1,0,0,0], w=[1,0,0,0]
        # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,1,0]
        # 4. u=[0,0,0,1], v=[0,-1,1,0], w=[0,0,0,1]
        # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
        # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[1,0,0,0]
        # 7. u=[0,-1,0,1], v=[0,0,1,1], w=[0,0,0,1]
        # Note: indices are flattened.
        # u index: (i,j) in 2x2.
        # v index: (j,k) in 2x2.
        # w index: (k,i) in 2x2.
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [0,1,0,1]),
            ([0,1,1,0], [1,0,0,0], [1,0,0,0]),
            ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
            ([0,0,0,1], [0,-1,1,0], [0,0,0,1]),
            ([1,1,0,0], [0,0,0,1], [0,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [1,0,0,0]),
            ([0,-1,0,1], [0,0,1,1], [0,0,0,1])
        ]
        
        for u, v, w in terms:
            for idx_u, val_u in enumerate(u):
                if val_u != 0:
                    i_local, j_local = divmod(idx_u, 2)
                    i = offset_i + i_local
                    j = offset_j + j_local
                    U[i*m + j][start_col] += val_u
            for idx_v, val_v in enumerate(v):
                if val_v != 0:
                    j_local, k_local = divmod(idx_v, 2)
                    j = offset_j + j_local
                    k = offset_k + k_local
                    V[j*p + k][start_col] += val_v
            for idx_w, val_w in enumerate(w):
                if val_w != 0:
                    k_local, i_local = divmod(idx_w, 2)
                    k = offset_k + k_local
                    i = offset_i + i_local
                    W[k*n + i][start_col] += val_w
            start_col += 1
        return start_col

    # Apply Strassen to the 4x4x4 part
    # A1 is rows 0-3, cols 0-3.
    # B1a is rows 0-3, cols 0-3.
    # C1a is rows 0-3, cols 0-3.
    
    # The 7 Strassen block products for 4x4x4:
    # Each block is 2x2.
    # Offsets for A blocks: (0,0), (0,2), (2,0), (2,2)
    # Offsets for B blocks: (0,0), (0,2), (2,0), (2,2)
    
    # M1 = (A11+A22)(B11+B22) -> This is a sum of 4 products?
    # No, Strassen computes M1 as a single product of two 2x2 matrices.
    # So we call add_strassen_2x2x2 for the block that represents M1?
    # No, M1 is the product of two specific 2x2 matrices.
    # The Strassen algorithm for 4x4x4 uses 7 products of 2x2 matrices.
    # Let's identify the 7 products.
    
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # Each Mk is a 2x2 matrix product.
    # We can decompose each Mk into 7 rank-1 terms.
    
    # However, the "A11+A22" is not a single 2x2 block in our U matrix.
    # It is a linear combination of two blocks.
    # Our U matrix has entries for each (i,j).
    # So for M1, the U part involves indices from A11 and A22.
    # We can handle this by adding the terms for A11 and A22 with appropriate signs/coefficients.
    
    # Let's refine the add_strassen_2x2x2 to accept a list of (offset, coeff) for U, V, W.
    
    def add_strassen_term(u_blocks, v_blocks, w_blocks, start_col):
        # u_blocks: list of (offset_i, offset_j, coeff)
        # v_blocks: list of (offset_j, offset_k, coeff)
        # w_blocks: list of (offset_k, offset_i, coeff)
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [0,1,0,1]),
            ([0,1,1,0], [1,0,0,0], [1,0,0,0]),
            ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
            ([0,0,0,1], [0,-1,1,0], [0,0,0,1]),
            ([1,1,0,0], [0,0,0,1], [0,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [1,0,0,0]),
            ([0,-1,0,1], [0,0,1,1], [0,0,0,1])
        ]
        
        for u, v, w in terms:
            for idx_u, val_u in enumerate(u):
                if val_u != 0:
                    i_local, j_local = divmod(idx_u, 2)
                    for off_i, off_j, coeff in u_blocks:
                        i = off_i + i_local
                        j = off_j + j_local
                        U[i*m + j][start_col] += val_u * coeff
            for idx_v, val_v in enumerate(v):
                if val_v != 0:
                    j_local, k_local = divmod(idx_v, 2)
                    for off_j, off_k, coeff in v_blocks:
                        j = off_j + j_local
                        k = off_k + k_local
                        V[j*p + k][start_col] += val_v * coeff
            for idx_w, val_w in enumerate(w):
                if val_w != 0:
                    k_local, i_local = divmod(idx_w, 2)
                    for off_k, off_i, coeff in w_blocks:
                        k = off_k + k_local
                        i = off_i + i_local
                        W[k*n + i][start_col] += val_w * coeff
            start_col += 1
        return start_col

    # Define the 7 Strassen terms for 4x4x4
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B blocks: B11(0,0), B12(0,2), B21(2,0), B22(2,2)
    # C blocks: C11(0,0), C12(0,2), C21(2,0), C22(2,2)
    
    # M1 = (A11+A22) * (B11+B22)
    u1 = [(0,0,1), (2,2,1)]
    v1 = [(0,0,1), (2,2,1)]
    w1 = [(0,0,1), (2,2,1)] # This is not right. W is the result.
    # The result C11, C12, C21, C22 are combinations of M1..M7.
    # We don't add W blocks directly in the term definition.
    # The term definition defines the tensor T = sum u_k v_k w_k.
    # The w_k vectors are fixed in the Strassen basis.
    # The w_blocks should be the identity? No.
    # The w part of the tensor is the C matrix.
    # In the Strassen decomposition, the C matrix is reconstructed from M1..M7.
    # But we are decomposing the *product* M1.
    # M1 is a 2x2 matrix. Its entries contribute to C11, C12, C21, C22.
    # Specifically, M1 contributes to all four C blocks.
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1?
    # No.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # So when we add the terms for M1, we must add them to the W matrix with coefficients corresponding to the C blocks.
    # The w vector in the Strassen term is for the 2x2 M matrix.
    # Let M be the 2x2 result.
    # M11 contributes to C11, C12, C21, C22?
    # No, M1 is a 2x2 matrix.
    # M1[0,0] contributes to C11[0,0], C11[0,1], C11[1,0], C11[1,1]?
    # No, C is 4x4.
    # M1 is the product of two 2x2 matrices.
    # M1 is added to C11, C12, C21, C22 with specific signs.
    
    # So for each term in the decomposition of M1, we add it to W at the locations of C11, C12, C21, C22.
    
    # Let's generalize add_strassen_term to take w_blocks as a list of (offset_k, offset_i, coeff).
    # For M1, w_blocks will have entries for C11, C12, C21, C22 all with coeff 1 (or -1 etc).
    
    # M1:
    u1 = [(0,0,1), (2,2,1)]
    v1 = [(0,0,1), (2,2,1)]
    w1 = [(0,0,1), (0,2,1), (2,0,1), (2,2,1)]
    col = add_strassen_term(u1, v1, w1, col)
    
    # M2 = (A21+A22) * B11
    u2 = [(2,0,1), (2,2,1)]
    v2 = [(0,0,1)]
    w2 = [(2,0,1), (2,2,1)] # M2 contributes to C21 and C22
    col = add_strassen_term(u2, v2, w2, col)
    
    # M3 = A11 * (B12-B22)
    u3 = [(0,0,1)]
    v3 = [(0,2,1), (2,2,-1)]
    w3 = [(0,2,1), (2,2,1)] # M3 contributes to C12 and C22
    col = add_strassen_term(u3, v3, w3, col)
    
    # M4 = A22 * (B21-B11)
    u4 = [(2,2,1)]
    v4 = [(2,0,1), (0,0,-1)]
    w4 = [(0,0,1), (2,0,1)] # M4 contributes to C11 and C21
    col = add_strassen_term(u4, v4, w4, col)
    
    # M5 = (A11+A12) * B22
    u5 = [(0,0,1), (0,2,1)]
    v5 = [(2,2,1)]
    w5 = [(0,0,1), (0,2,1)] # M5 contributes to C11 and C12
    col = add_strassen_term(u5, v5, w5, col)
    
    # M6 = (A21-A11) * (B11+B12)
    u6 = [(2,0,1), (0,0,-1)]
    v6 = [(0,0,1), (0,2,1)]
    w6 = [(2,2,1)] # M6 contributes to C22
    col = add_strassen_term(u6, v6, w6, col)
    
    # M7 = (A12-A22) * (B21+B22)
    u7 = [(0,2,1), (2,2,-1)]
    v7 = [(2,0,1), (2,2,1)]
    w7 = [(0,0,1), (2,2,1)] # M7 contributes to C11 and C22
    col = add_strassen_term(u7, v7, w7, col)
    
    # Part 2: A1 B1b (4x4 x 2)
    # A1 is 4x4. B1b is 4x2 (cols 4,5 of B).
    # Schoolbook.
    # i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    # Part 3: A2 B2 (4x1 x 6)
    # A2 is 4x1 (col 4 of A). B2 is 1x6 (row 4 of B).
    # Schoolbook.
    # i in 0..3, j in 4..4, k in 0..5
    for i in range(4):
        for j in range(4, 5):
            for k in range(6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
