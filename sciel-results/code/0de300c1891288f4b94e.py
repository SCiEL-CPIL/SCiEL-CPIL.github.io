import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 105
    # We will build U, V, W lists of lists
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j), v_idx is (j, k), w_idx is (k, i)
        # But we need to map to flat indices
        # u_flat = i * m + j
        # v_flat = j * p + k
        # w_flat = k * n + i
        
        # We store columns as lists of length n*m, m*p, p*n
        # It's more efficient to build the full matrices at the end
        # Or build sparse representations
        
        pass

    # Let's build the full matrices with zeros and fill them
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (4x4) * B_TL (4x4) -> C_TL (4x4)
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # This is M_{4,4,4}. Rank 49.
    # We need the Strassen decomposition for 4x4x4.
    # We can generate it by applying Strassen 2x2x2 to the 2x2 blocks of the 4x4 matrix.
    # A 4x4 matrix has 4 blocks of 2x2.
    # Strassen on 2x2 blocks uses 7 multiplications.
    # Total 4 * 7 = 28? No.
    # Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # This is for 2x2 blocks.
    # For 4x4, we treat it as 2x2 blocks of 2x2 matrices.
    # So we need 7 "multiplications" of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So total rank 7 * 7 = 49.
    
    # We will implement a generic Strassen generator for 2^k x 2^k.
    def get_strassen_2x2():
        # Returns list of (u_factor, v_factor, w_factor) for 2x2x2
        # u_factor, v_factor, w_factor are 2x2 matrices (or sparse representations)
        # Actually, we just need the coefficients.
        # The tensor is sum of 7 rank-1 terms.
        # Each term is (A_part * B_part) added to C_part.
        # We need to express A_part and B_part as linear combinations of A_ij and B_ij.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M6 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M6
        
        # We need to map this to the tensor indices.
        # This is getting complex to code dynamically.
        # Given the time limit, I will hardcode the 49 terms for 4x4x4 Strassen
        # or use a known list.
        
        # Alternatively, I can just use the schoolbook for the 4x4x4 part if Strassen is too hard to code quickly.
        # Schoolbook rank is 64.
        # 64 + 32 + 24 = 120. No improvement.
        # I need Strassen.
        
        # Let's try to generate the 49 terms programmatically.
        # A 4x4x4 tensor decomposition of rank 49.
        # I will use a recursive function to generate the Strassen terms for 2x2x2,
        # and then apply it to the 4x4 case by treating it as 2x2 blocks.
        
        pass

    # To be safe and fast, I will implement the 4x4x4 Strassen decomposition explicitly.
    # There are 49 terms.
    # I will write a helper that returns the 49 (i, j, k, coeff_u, coeff_v, coeff_w) tuples?
    # No, the decomposition is a sum of rank-1 tensors.
    # Each rank-1 tensor is defined by vectors u, v, w.
    # For Strassen, the vectors are not just single indices, but sums.
    # However, the tensor decomposition definition requires U, V, W matrices.
    # Each column c corresponds to a rank-1 term.
    # U[:, c] is the u vector, V[:, c] is the v vector, W[:, c] is the w vector.
    # For Strassen, the u vector for a term like (A11+A22) will have 1s at positions corresponding to A11 and A22.
    # So U is a matrix with entries 0, 1, -1.
    
    # I will generate the 49 columns for the 4x4x4 part.
    
    # Part 2: A_L (4x4) * B_TR (4x2) -> C_TR (4x2)
    # Standard multiplication. Rank 32.
    # i in 0..3, j in 0..3, k in 4..5 (since B is 4x6, cols 0..3 are TL, 4..5 are TR)
    # Wait, B is 5x6.
    # B_T is top 4 rows, all 6 cols.
    # B_TL is cols 0..3. B_TR is cols 4..5.
    # So for Part 2, k is 4,5.
    
    # Part 3: A_R (4x1) * B_B (1x6) -> C_B (4x6)
    # A_R is col 4 of A.
    # B_B is row 4 of B.
    # i in 0..3, j=4, k in 0..5.
    # Rank 24.
    
    # Total rank 49 + 32 + 24 = 105.
    
    # Implementation details:
    # I need the Strassen 4x4x4 decomposition.
    # I will hardcode the logic to generate the 49 terms.
    
    # Strassen 2x2x2 terms (7 terms):
    # Each term t has:
    # u_t: 2x2 matrix (coeffs for A)
    # v_t: 2x2 matrix (coeffs for B)
    # w_t: 2x2 matrix (coeffs for C)
    
    # For 4x4x4, we have 4 blocks of A: A11, A12, A21, A22 (each 2x2).
    # Similarly for B and C.
    # We apply Strassen to the blocks.
    # This creates 7 "block multiplications".
    # Each block multiplication is a 2x2x2 tensor of rank 7.
    # So we expand each of the 7 block terms into 7 scalar terms.
    # Total 49 terms.
    
    # I will write a function to get the 7 Strassen 2x2x2 block definitions.
    
    def strassen_2x2_blocks():
        # Returns list of 7 tuples (u_block, v_block, w_block)
        # u_block, v_block, w_block are 2x2 matrices of integers
        terms = []
        # M1 = (A11+A22)(B11+B22)
        u1 = [[1, 0], [0, 1]]
        v1 = [[1, 0], [0, 1]]
        w1 = [[1, 0], [0, 1]] # C11 += M1, C22 += M1
        # Wait, C11 and C22 get M1.
        # So w1 should have 1 at (0,0) and (1,1)?
        # No, the tensor term is u \otimes v \otimes w.
        # If w has 1 at (0,0), it adds to C11.
        # If w has 1 at (1,1), it adds to C22.
        # So for M1, we need two terms? Or one term with w having two 1s?
        # The rank-1 tensor is defined by vectors.
        # If w is a vector with two 1s, it's still rank 1?
        # Yes, if u and v are fixed, and w is a sum of basis vectors, it's a sum of rank-1 tensors.
        # But we want to minimize rank.
        # Strassen's algorithm is a sum of 7 rank-1 tensors.
        # Each rank-1 tensor corresponds to one multiplication.
        # The result is added to specific C blocks.
        # So for M1, we have one multiplication.
        # The result is added to C11 and C22.
        # This means the term contributes to C11 and C22.
        # In the tensor decomposition, this is represented by a single column in U, V, W?
        # No, if the same multiplication result is added to two places, it's still one rank-1 term in the tensor?
        # $T = \sum u_k \otimes v_k \otimes w_k$.
        # If $w_k$ has non-zero entries at multiple positions, it's fine.
        # So for M1, $w_k$ has 1 at (0,0) and 1 at (1,1).
        
        # Let's define the 7 terms for 2x2x2.
        # u, v, w are 2x2 matrices.
        
        # M1
        u = [[1, 0], [0, 1]]
        v = [[1, 0], [0, 1]]
        w = [[1, 0], [0, 1]]
        terms.append((u, v, w))
        
        # M2
        u = [[0, 0], [1, 1]]
        v = [[1, 0], [0, 0]]
        w = [[0, 0], [1, 0]] # C21 += M2
        terms.append((u, v, w))
        
        # M3
        u = [[1, 0], [0, 0]]
        v = [[0, 1], [0, -1]]
        w = [[0, 1], [0, 0]] # C12 += M3
        terms.append((u, v, w))
        
        # M4
        u = [[0, 0], [0, 1]]
        v = [[0, 1], [-1, 0]]
        w = [[-1, 0], [0, 0]] # C11 -= M4, C21 += M4
        # Wait, C11 = M1 - M4 + M6 + M7.
        # C21 = M2 + M4.
        # So M4 contributes -1 to C11 and +1 to C21.
        w = [[-1, 0], [1, 0]]
        terms.append((u, v, w))
        
        # M5
        u = [[1, 1], [0, 0]]
        v = [[0, 0], [0, 1]]
        w = [[0, 1], [0, 1]] # C12 += M5, C22 -= M5?
        # C12 = M3 + M5.
        # C22 = M1 - M3 + M5 - M6.
        # So M5 contributes +1 to C12 and +1 to C22.
        w = [[0, 1], [0, 1]]
        terms.append((u, v, w))
        
        # M6
        u = [[-1, 0], [1, 0]]
        v = [[1, 1], [0, 0]]
        w = [[1, 0], [0, -1]] # C11 += M6, C22 -= M6
        terms.append((u, v, w))
        
        # M7
        u = [[0, 1], [-1, 0]]
        v = [[0, 0], [1, 1]]
        w = [[1, 0], [0, 1]] # C11 += M7, C22 += M7
        terms.append((u, v, w))
        
        return terms

    # Now, for 4x4x4, we apply this to the 2x2 blocks.
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # For each of the 7 Strassen terms, we have u_block, v_block, w_block.
    # u_block is a 2x2 matrix of coefficients for the A blocks.
    # e.g., u_block[0][0] is coeff for A11.
    # We need to expand this into the 4x4 U vector.
    # The U vector for the 4x4 case is 16-dimensional.
    # For a given Strassen term t, and a given block position (bi, bj) in the 2x2 block grid,
    # the coefficient for the scalar A_{i,j} is u_block[bi][bj] * (scalar coeff from the inner 2x2 Strassen).
    
    # Wait, the inner 2x2 Strassen is applied to the scalar elements of the blocks?
    # No, the blocks are 2x2 matrices.
    # The Strassen algorithm on blocks uses 7 matrix multiplications.
    # Each matrix multiplication is a 2x2x2 tensor of rank 7.
    # So for each of the 7 block terms, we generate 7 scalar terms.
    # Total 49 terms.
    
    # For each block term t (0..6):
    #   For each inner term s (0..6):
    #     The u vector is formed by combining u_block[t] and u_inner[s].
    #     Specifically, if u_block[t][bi][bj] is non-zero, and u_inner[s][i2][j2] is non-zero,
    #     then the scalar A at row (bi*2 + i2) and col (bj*2 + j2) has coefficient
    #     u_block[t][bi][bj] * u_inner[s][i2][j2].
    
    # I will implement this logic.

    strassen_2x2 = strassen_2x2_blocks()
    
    # Generate 49 terms for 4x4x4
    for t in range(7):
        u_block, v_block, w_block = strassen_2x2[t]
        for s in range(7):
            u_inner, v_inner, w_inner = strassen_2x2[s]
            
            # Build U vector (size 16)
            u_vec = [0] * 16
            for bi in range(2):
                for bj in range(2):
                    coeff_block = u_block[bi][bj]
                    if coeff_block != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                coeff_inner = u_inner[i2][j2]
                                if coeff_inner != 0:
                                    i = bi * 2 + i2
                                    j = bj * 2 + j2
                                    u_vec[i * 4 + j] += coeff_block * coeff_inner
                                    
            # Build V vector (size 16)
            v_vec = [0] * 16
            for bi in range(2):
                for bj in range(2):
                    coeff_block = v_block[bi][bj]
                    if coeff_block != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                coeff_inner = v_inner[i2][j2]
                                if coeff_inner != 0:
                                    i = bi * 2 + i2
                                    j = bj * 2 + j2
                                    v_vec[i * 4 + j] += coeff_block * coeff_inner
                                    
            # Build W vector (size 16)
            w_vec = [0] * 16
            for bi in range(2):
                for bj in range(2):
                    coeff_block = w_block[bi][bj]
                    if coeff_block != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                coeff_inner = w_inner[i2][j2]
                                if coeff_inner != 0:
                                    i = bi * 2 + i2
                                    j = bj * 2 + j2
                                    w_vec[i * 4 + j] += coeff_block * coeff_inner
                                    
            # Add to global U, V, W
            # Indices for this part: i in 0..3, j in 0..3, k in 0..3
            # U index: i * 5 + j (since m=5)
            # V index: j * 6 + k (since p=6)
            # W index: k * 4 + i (since n=4)
            
            for idx in range(16):
                u_val = u_vec[idx]
                if u_val != 0:
                    i = idx // 4
                    j = idx % 4
                    U[i * 5 + j][col] = u_val
                    
                v_val = v_vec[idx]
                if v_val != 0:
                    j = idx // 4
                    k = idx % 4
                    V[j * 6 + k][col] = v_val
                    
                w_val = w_vec[idx]
                if w_val != 0:
                    k = idx // 4
                    i = idx % 4
                    W[k * 4 + i][col] = w_val
                    
            col += 1
            
    # Part 2: A_L (4x4) * B_TR (4x2)
    # i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # Part 3: A_R (4x1) * B_B (1x6)
    # i in 0..3, j=4, k in 0..5
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][col] = 1
            V[4 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
