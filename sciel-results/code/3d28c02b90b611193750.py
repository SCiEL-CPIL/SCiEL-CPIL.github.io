DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # We'll collect terms as lists of (u_indices, v_indices, w_indices, coeffs)
    # Each term contributes to a specific set of positions in U, V, W.
    # A term is a rank-1 tensor: U[:,r] = u, V[:,r] = v, W[:,r] = w
    # where u, v, w are vectors with specific non-zero entries.
    
    terms = []  # list of (u_dict, v_dict, w_dict) where dict maps index -> value
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen's 2x2x2 multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 7 products:
    # P1 = (a11 + a22)(b11 + b22)
    # P2 = (a21 + a22)b11
    # P3 = a11(b12 - b22)
    # P4 = a22(b21 - b11)
    # P5 = (a11 + a12)b22
    # P6 = (a21 - a11)(b11 + b12)
    # P7 = (a12 - a22)(b21 + b22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For a 2x2x2 block at position (i0, j0, k0) in the global matrices:
    # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
    
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 multiplication at block (i0,j0,k0)."""
        # A elements: a11=(i0,j0), a12=(i0,j0+1), a21=(i0+1,j0), a22=(i0+1,j0+1)
        # B elements: b11=(j0,k0), b12=(j0,k0+1), b21=(j0+1,k0), b22=(j0+1,k0+1)
        # C elements: c11=(i0,k0), c12=(i0,k0+1), c21=(i0+1,k0), c22=(i0+1,k0+1)
        
        a11 = (i0, j0)
        a12 = (i0, j0+1)
        a21 = (i0+1, j0)
        a22 = (i0+1, j0+1)
        
        b11 = (j0, k0)
        b12 = (j0, k0+1)
        b21 = (j0+1, k0)
        b22 = (j0+1, k0+1)
        
        c11 = (i0, k0)
        c12 = (i0, k0+1)
        c21 = (i0+1, k0)
        c22 = (i0+1, k0+1)
        
        # P1 = (a11 + a22)(b11 + b22)
        u1 = {a11*m + a11[1]: 1, (a22[0]*m + a22[1]): 1}  # wait, a11 is (i,j) tuple
        # Let me fix: u index is i*m + j
        u1 = {a11[0]*m + a11[1]: 1, a22[0]*m + a22[1]: 1}
        v1 = {b11[0]*p + b11[1]: 1, b22[0]*p + b22[1]: 1}
        w1 = {c11[0]*n + c11[1]: 1, c22[0]*n + c22[1]: 1}
        add_term(u1, v1, w1)
        
        # P2 = (a21 + a22)b11
        u2 = {a21[0]*m + a21[1]: 1, a22[0]*m + a22[1]: 1}
        v2 = {b11[0]*p + b11[1]: 1}
        w2 = {c21[0]*n + c21[1]: 1, c22[0]*n + c22[1]: 1}
        add_term(u2, v2, w2)
        
        # P3 = a11(b12 - b22)
        u3 = {a11[0]*m + a11[1]: 1}
        v3 = {b12[0]*p + b12[1]: 1, b22[0]*p + b22[1]: -1}
        w3 = {c12[0]*n + c12[1]: 1, c22[0]*n + c22[1]: 1}
        add_term(u3, v3, w3)
        
        # P4 = a22(b21 - b11)
        u4 = {a22[0]*m + a22[1]: 1}
        v4 = {b21[0]*p + b21[1]: 1, b11[0]*p + b11[1]: -1}
        w4 = {c11[0]*n + c11[1]: 1, c21[0]*n + c21[1]: 1}
        add_term(u4, v4, w4)
        
        # P5 = (a11 + a12)b22
        u5 = {a11[0]*m + a11[1]: 1, a12[0]*m + a12[1]: 1}
        v5 = {b22[0]*p + b22[1]: 1}
        w5 = {c11[0]*n + c11[1]: -1, c12[0]*n + c12[1]: 1}
        add_term(u5, v5, w5)
        
        # P6 = (a21 - a11)(b11 + b12)
        u6 = {a21[0]*m + a21[1]: 1, a11[0]*m + a11[1]: -1}
        v6 = {b11[0]*p + b11[1]: 1, b12[0]*p + b12[1]: 1}
        w6 = {c22[0]*n + c22[1]: 1}
        add_term(u6, v6, w6)
        
        # P7 = (a12 - a22)(b21 + b22)
        u7 = {a12[0]*m + a12[1]: 1, a22[0]*m + a22[1]: -1}
        v7 = {b21[0]*p + b21[1]: 1, b22[0]*p + b22[1]: 1}
        w7 = {c11[0]*n + c11[1]: 1}
        add_term(u7, v7, w7)
    
    # Schoolbook 2x2x1 multiplication: A (2x2) * B (2x1) = C (2x1)
    # C1 = a11*b1 + a12*b2
    # C2 = a21*b1 + a22*b2
    # 4 terms
    
    def schoolbook_2x2x1(i0, j0, k0):
        """Add 4 schoolbook terms for 2x2x1 multiplication at block (i0,j0,k0)."""
        # A: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # B: (j0,k0), (j0+1,k0)
        # C: (i0,k0), (i0+1,k0)
        
        a11 = (i0, j0)
        a12 = (i0, j0+1)
        a21 = (i0+1, j0)
        a22 = (i0+1, j0+1)
        
        b1 = (j0, k0)
        b2 = (j0+1, k0)
        
        c1 = (i0, k0)
        c2 = (i0+1, k0)
        
        # a11 * b1
        add_term({a11[0]*m + a11[1]: 1}, {b1[0]*p + b1[1]: 1}, {c1[0]*n + c1[1]: 1})
        # a12 * b2
        add_term({a12[0]*m + a12[1]: 1}, {b2[0]*p + b2[1]: 1}, {c1[0]*n + c1[1]: 1})
        # a21 * b1
        add_term({a21[0]*m + a21[1]: 1}, {b1[0]*p + b1[1]: 1}, {c2[0]*n + c2[1]: 1})
        # a22 * b2
        add_term({a22[0]*m + a22[1]: 1}, {b2[0]*p + b2[1]: 1}, {c2[0]*n + c2[1]: 1})
    
    # Schoolbook 2x2x2 is just 4 terms, but we use Strassen (7 terms) for efficiency
    
    # Now, the 4x4x4 part: decompose into four 2x2x2 blocks
    # Block (0,0,0): rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B; rows 0-1, cols 0-1 of C
    strassen_2x2x2(0, 0, 0)
    # Block (0,0,2): rows 0-1, cols 0-1 of A; rows 0-1, cols 2-3 of B; rows 0-1, cols 2-3 of C
    strassen_2x2x2(0, 0, 2)
    # Block (0,2,0): rows 0-1, cols 2-3 of A; rows 0-1, cols 0-1 of B; rows 0-1, cols 0-1 of C
    strassen_2x2x2(0, 2, 0)
    # Block (0,2,2): rows 0-1, cols 2-3 of A; rows 0-1, cols 2-3 of B; rows 0-1, cols 2-3 of C
    strassen_2x2x2(0, 2, 2)
    # Block (2,0,0): rows 2-3, cols 0-1 of A; rows 0-1, cols 0-1 of B; rows 2-3, cols 0-1 of C
    strassen_2x2x2(2, 0, 0)
    # Block (2,0,2): rows 2-3, cols 0-1 of A; rows 0-1, cols 2-3 of B; rows 2-3, cols 2-3 of C
    strassen_2x2x2(2, 0, 2)
    # Block (2,2,0): rows 2-3, cols 2-3 of A; rows 0-1, cols 0-1 of B; rows 2-3, cols 0-1 of C
    strassen_2x2x2(2, 2, 0)
    # Block (2,2,2): rows 2-3, cols 2-3 of A; rows 0-1, cols 2-3 of B; rows 2-3, cols 2-3 of C
    strassen_2x2x2(2, 2, 2)
    
    # That's 8 blocks * 7 terms = 56 terms for the 4x4x4 part.
    # But wait, this is just tiling, not true Strassen on 4x4. 
    # True Strassen on 4x4 would give rank 7^2 = 49.
    # Let me implement proper Strassen on 4x4.
    
    # Clear terms and restart
    terms = []
    
    # Proper Strassen for 4x4x4:
    # Split A into 4 2x2 blocks: A11, A12, A21, A22
    # Split B into 4 2x2 blocks: B11, B12, B21, B22
    # C = A*B, split into 4 2x2 blocks: C11, C12, C21, C22
    # 
    # Strassen computes 7 products of 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2x2 multiplication, which we decompose with Strassen (7 terms each).
    # So total: 7 * 7 = 49 terms for 4x4x4.
    
    def strassen_4x4x4():
        """Decompose 4x4x4 matrix multiplication using Strassen, returning 49 rank-1 terms."""
        # Block positions:
        # A11: rows 0-1, cols 0-1 -> (0,0)
        # A12: rows 0-1, cols 2-3 -> (0,2)
        # A21: rows 2-3, cols 0-1 -> (2,0)
        # A22: rows 2-3, cols 2-3 -> (2,2)
        #
        # B11: rows 0-1, cols 0-1 -> (0,0)
        # B12: rows 0-1, cols 2-3 -> (0,2)
        # B21: rows 2-3, cols 0-1 -> (2,0)
        # B22: rows 2-3, cols 2-3 -> (2,2)
        #
        # C11: rows 0-1, cols 0-1 -> (0,0)
        # C12: rows 0-1, cols 2-3 -> (0,2)
        # C21: rows 2-3, cols 0-1 -> (2,0)
        # C22: rows 2-3, cols 2-3 -> (2,2)
        
        # M1 = (A11 + A22)(B11 + B22) -> C11, C22 contributions
        # The "A" part is A11+A22, which has non-zero entries in blocks (0,0) and (2,2)
        # The "B" part is B11+B22, which has non-zero entries in blocks (0,0) and (2,2)
        # The result contributes to C11 (block 0,0) and C22 (block 2,2)
        
        # For M1, we need to decompose the product of (A11+A22) and (B11+B22).
        # A11+A22 is a 4x4 matrix with blocks:
        #   [A11, 0; 0, A22]
        # B11+B22 is a 4x4 matrix with blocks:
        #   [B11, 0; 0, B22]
        # Their product is:
        #   [A11*B11, 0; 0, A22*B22]
        # This contributes to C11 (top-left 2x2) and C22 (bottom-right 2x2).
        
        # So M1 decomposes into two 2x2x2 Strassen decompositions:
        # One for A11*B11 -> C11
        # One for A22*B22 -> C22
        
        # But wait, that's not quite right. The Strassen formula for M1 uses the
        # combined A and B, and the 2x2x2 decomposition should handle the full 2x2x2
        # for each block. Let me think again.
        
        # Actually, M1 = (A11+A22)(B11+B22) is a 4x4 matrix. To decompose it into
        # rank-1 tensors, we can view it as:
        # M1 = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But that's 4 products, not 1. The Strassen trick is that we compute
        # (A11+A22) and (B11+B22) first, then multiply. The multiplication
        # (A11+A22)(B11+B22) is a 4x4x4 multiplication, but with the special
        # structure that A and B are block-diagonal with 2x2 blocks.
        
        # Hmm, this is getting complicated. Let me just use the tiling approach
        # which gives 56 terms, and accept that it's not optimal but better than 112.
        
        pass
    
    # Let me just use the tiling approach: 8 blocks of 2x2x2, each with Strassen (7 terms)
    # Total: 56 terms for 4x4x4.
    
    # For the 4x4x3 part:
    # A is 4x4, B is 4x3, C is 4x3.
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B into 2x1 blocks: B1 (rows 0-1, col 0), B2 (rows 0-1, cols 1-2), etc.
    # Actually, B is 4x3. Split into:
    # B_top: rows 0-1, all 3 cols -> 2x3
    # B_bot: rows 2-3, all 3 cols -> 2x3
    # 
    # C = A*B, split into:
    # C_top: rows 0-1, all 3 cols -> 2x3
    # C_bot: rows 2-3, all 3 cols -> 2x3
    #
    # C_top = A11*B_top + A12*B_bot  (since A11 is rows 0-1, cols 0-1; A12 is rows 0-1, cols 2-3)
    # Wait, A is 4x4, B is 4x3. 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For i in 0-1 (top half of C):
    #   C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    #   = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    #   = (A_top * B_top)[i,k] + (A_top * B_bot)[i,k]
    # where A_top = A[0:2, :], B_top = B[0:2, :], B_bot = B[2:4, :]
    #
    # Similarly for bottom half.
    
    # Let me just use schoolbook for the 4x4x3 part to keep it simple.
    # 4*4*3 = 48 terms.
    # Total: 56 + 48 = 104. That's worse than 112? No, 104 < 112.
    
    # Actually, let me reconsider. The tiling approach for 4x4x4 gives 56 terms.
    # For 4x4x3, I can tile into 2x2x1 and 2x2x2 blocks.
    # B is 4x3. Split into:
    # B[:,0:1] (4x1), B[:,1:3] (4x2)
    # Or B[0:2, :] (2x3), B[2:4, :] (2x3)
    
    # Let me tile 4x4x3 into:
    # 4 blocks of 2x2x3? No, 2x2x3 isn't standard.
    # 
    # Let me just do: for each of the 4 2x2 blocks of A, multiply by the full 2x3 or 4x3 block of B.
    # 
    # Simpler: just use schoolbook for 4x4x3: 48 terms.
    # Total: 56 + 48 = 104.
    
    # But I can do better for 4x4x3. Let me tile it.
    # A is 4x4, B is 4x3, C is 4x3.
    # Split A into 2x2 blocks: A11(0-1,0-1), A12(0-1,2-3), A21(2-3,0-1), A22(2-3,2-3)
    # Split B into 2x3 blocks: B1(0-1,:), B2(2-3,:)
    # Split C into 2x3 blocks: C1(0-1,:), C2(2-3,:)
    #
    # C1 = A11*B1 + A12*B2  (2x3 = 2x2*2x3 + 2x2*2x3)
    # C2 = A21*B1 + A22*B2
    #
    # Each 2x2*2x3 multiplication: 
    #   C = A*B where A is 2x2, B is 2x3.
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   This is 2*2*3 = 12 scalar products, but we can use Strassen-like tricks.
    #   Actually, 2x2x3 can be done in 7 terms? Let me check.
    #   The rank of 2x2x3 matrix multiplication is 7 (same as 2x2x2 since 3>2 doesn't help).
    #   Actually, I think it's 7.
    
    # For 2x2x3: A(2x2) * B(2x3) = C(2x3)
    # C = [[a11*b11 + a12*b21, a11*b12 + a12*b22, a11*b13 + a12*b23],
    #      [a21*b11 + a22*b21, a21*b12 + a22*b22, a21*b13 + a22*b23]]
    # 
    # This is like doing 2x2x2 twice (for cols 0-1 and col 2 separately) but that's 14 terms.
    # Or we can be smarter.
    
    # Let me just use schoolbook for 2x2x3: 4*3 = 12 terms per block.
    # 4 blocks * 12 = 48 terms for 4x4x3.
    # Total: 56 + 48 = 104.
    
    # OK let me just implement this.
    
    # 4x4x4 part: 8 Strassen 2x2x2 blocks = 56 terms
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    strassen_2x2x2(0, 2, 0)
    strassen_2x2x2(0, 2, 2)
    strassen_2x2x2(2, 0, 0)
    strassen_2x2x2(2, 0, 2)
    strassen_2x2x2(2, 2, 0)
    strassen_2x2x2(2, 2, 2)
    
    # 4x4x3 part: schoolbook
    # A is 4x4 (indices i=0..3, j=0..3)
    # B is 4x3 (indices j=0..3, k=0..2)
    # C is 4x3 (indices i=0..3, k=0..2)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = {i*m + j: 1}
                v = {j*p + k: 1}
                w = {k*n + i: 1}
                add_term(u, v, w)
    
    # Total terms: 56 + 48 = 104
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
