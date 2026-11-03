"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x7."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll use a block decomposition approach with Strassen for 2x2 blocks
    # 
    # A is 3x4, B is 4x7, C is 3x7
    # 
    # Split A rows: [2, 1], A cols: [2, 2]
    # Split B rows: [2, 2], B cols: [2, 2, 2, 1]
    #
    # A = [[A11, A12], [A21, A22]] where A11,A12,A21 are 2x2, A22 is 1x2
    # Wait, let me be more careful.
    #
    # A (3x4):
    #   rows 0-1, cols 0-1: A11 (2x2)
    #   rows 0-1, cols 2-3: A12 (2x2)  
    #   row 2, cols 0-1: A21 (1x2)
    #   row 2, cols 2-3: A22 (1x2)
    #
    # B (4x7):
    #   rows 0-1, cols 0-1: B11 (2x2)
    #   rows 0-1, cols 2-3: B12 (2x2)
    #   rows 0-1, cols 4-5: B13 (2x2)
    #   rows 0-1, col 6: B14 (2x1)
    #   rows 2-3, cols 0-1: B21 (2x2)
    #   rows 2-3, cols 2-3: B22 (2x2)
    #   rows 2-3, cols 4-5: B23 (2x2)
    #   rows 2-3, col 6: B24 (2x1)
    #
    # C (3x7):
    #   rows 0-1, cols 0-1: C11 = A11*B11 + A12*B21 (2x2)
    #   rows 0-1, cols 2-3: C12 = A11*B12 + A12*B22 (2x2)
    #   rows 0-1, cols 4-5: C13 = A11*B13 + A12*B23 (2x2)
    #   rows 0-1, col 6: C14 = A11*B14 + A12*B24 (2x1)
    #   row 2, cols 0-1: C21 = A21*B11 + A22*B21 (1x2)
    #   row 2, cols 2-3: C22 = A21*B12 + A22*B22 (1x2)
    #   row 2, cols 4-5: C23 = A21*B13 + A22*B23 (1x2)
    #   row 2, col 6: C24 = A21*B14 + A22*B24 (1x1)
    #
    # For 2x2 x 2x2 products, use Strassen (7 mults each):
    # C11: A11*B11 + A12*B21 -> use Strassen on the sum? No, Strassen works on single products.
    # Actually, we need to compute A11*B11 and A12*B21 separately, then add.
    # Each 2x2*2x2 needs 4 mults schoolbook or 7 with Strassen... but Strassen is for computing
    # a single 2x2 product. For the sum of two products, we can't directly apply Strassen.
    #
    # Let me reconsider. The tensor decomposition approach:
    # Each "multiplication" in the decomposition corresponds to one rank-1 term.
    # 
    # Schoolbook: 3*4*7 = 84 terms
    # 
    # Let me try a different splitting:
    # Split m=4 into 2+2, and use the fact that for 2x2 x 2x2, Strassen gives rank 7.
    #
    # Actually, let me think about this differently. The matrix multiplication tensor 
    # for n x m times m x p has rank at most n*m*p (schoolbook). 
    #
    # For 3x4 x 4x7, let me try to use the structure more cleverly.
    #
    # One approach: split p=7 into 3+4
    # C = A * B where B = [B_left (4x3) | B_right (4x4)]
    # C_left (3x3) = A * B_left
    # C_right (3x4) = A * B_right
    #
    # For 3x3 x 3x3: known minimal rank is 21 (Strassen-like)
    # For 3x4 x 4x4: need to figure out
    #
    # Actually, let me just try to find a good decomposition numerically or use
    # known results. The key insight from the inspiration is to use the 3x4x3 
    # decomposition with rank 21.
    #
    # Let me try: split p=7 as 3+4
    # C[:, :3] = A @ B[:, :3]  (3x3 output)
    # C[:, 3:] = A @ B[:, 3:]  (3x4 output)
    #
    # For the 3x3 part: use rank-21 decomposition (known optimal for 3x3x3)
    # For the 3x4 part: use schoolbook 3*4*4 = 48? Or can we do better?
    #
    # Actually, let me try a completely different approach. Let me just try to 
    # find a low-rank decomposition by exploiting the linear structure.
    #
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Let me try rank = 50 and see if I can find a solution via optimization.
    # But that might be too slow. Let me think of a constructive approach.
    
    # Constructive approach using block decomposition:
    # 
    # Split A (3x4) into two parts: A_top (2x4) and A_bot (1x4)
    # Split B (4x7) into two parts: B_left (4x3) and B_right (4x4)
    # 
    # C_top_left (2x3) = A_top @ B_left
    # C_top_right (2x4) = A_top @ B_right  
    # C_bot_left (1x3) = A_bot @ B_left
    # C_bot_right (1x4) = A_bot @ B_right
    #
    # For 2x4 x 4x3: rank at most 2*4*3 = 24, but can use Strassen-like
    # For 2x4 x 4x4: rank at most 2*4*4 = 32
    # For 1x4 x 4x3: rank at most 1*4*3 = 12
    # For 1x4 x 4x4: rank at most 1*4*4 = 16
    # Total: 24 + 32 + 12 + 16 = 84 (schoolbook)
    #
    # To improve, I need to use the fact that 2x2 x 2x2 has rank 7 (Strassen)
    # instead of 4 (schoolbook).
    #
    # Let me split further:
    # A_top (2x4) -> A_top_left (2x2), A_top_right (2x2)
    # B_left (4x3) -> B_top_left (2x3), B_bot_left (2x3)
    # B_right (4x4) -> B_top_right (2x4), B_bot_right (2x4)
    #
    # C_top_left = A_top_left @ B_top_left + A_top_right @ B_bot_left
    # C_top_right = A_top_left @ B_top_right + A_top_right @ B_bot_right
    #
    # Each term is 2x2 x 2x3 or 2x2 x 2x4
    #
    # 2x2 x 2x3: rank at most 12, but we can use Strassen on the 2x2 part
    # Actually for 2x2 x 2xk, the rank is at most 2*2*k = 4k schoolbook
    # With Strassen: 7*k? No, Strassen gives 7 mults for 2x2 x 2x2.
    # For 2x2 x 2xk, we can split k into 2s and use Strassen: 7*ceil(k/2)
    #
    # This is getting complex. Let me just try to construct a specific low-rank
    # decomposition using known Strassen formulas.
    
    # Let me use a direct construction:
    # 
    # For the 2x2 x 2x2 Strassen decomposition, the 7 multiplications are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Now let me build the full decomposition.
    # 
    # I'll use the following block structure:
    # A (3x4): split rows as [2,1], cols as [2,2]
    # B (4x7): split rows as [2,2], cols as [2,2,2,1]
    #
    # Let me define the blocks:
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2:4] (2x2)
    # A21 = A[2:3, 0:2] (1x2)
    # A22 = A[2:3, 2:4] (1x2)
    #
    # B11 = B[0:2, 0:2] (2x2)
    # B12 = B[0:2, 2:4] (2x2)
    # B13 = B[0:2, 4:6] (2x2)
    # B14 = B[0:2, 6:7] (2x1)
    # B21 = B[2:4, 0:2] (2x2)
    # B22 = B[2:4, 2:4] (2x2)
    # B23 = B[2:4, 4:6] (2x2)
    # B24 = B[2:4, 6:7] (2x1)
    #
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C13 = A11*B13 + A12*B23 (2x2)
    # C14 = A11*B14 + A12*B24 (2x1)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x2)
    # C23 = A21*B13 + A22*B23 (1x2)
    # C24 = A21*B14 + A22*B24 (1x1)
    #
    # For the 2x2 x 2x2 products, use Strassen (7 mults each).
    # For 2x2 x 2x1 products, use 4 mults (schoolbook, since Strassen doesn't help for non-square).
    # For 1x2 x 2x2 products, use 4 mults.
    # For 1x2 x 2x1 products, use 2 mults.
    #
    # C11: A11*B11 (Strassen: 7) + A12*B21 (Strassen: 7) = 14
    # C12: A11*B12 (Strassen: 7) + A12*B22 (Strassen: 7) = 14
    # C13: A11*B13 (Strassen: 7) + A12*B23 (Strassen: 7) = 14
    # C14: A11*B14 (4) + A12*B24 (4) = 8
    # C21: A21*B11 (4) + A22*B21 (4) = 8
    # C22: A21*B12 (4) + A22*B22 (4) = 8
    # C23: A21*B13 (4) + A22*B23 (4) = 8
    # C24: A21*B14 (2) + A22*B24 (2) = 4
    #
    # Total: 14+14+14+8+8+8+8+4 = 78
    #
    # That's better than 84! Let me construct this.
    
    # Build the decomposition
    U_list = []  # Will be list of (i*m+j, value) pairs per term
    V_list = []  # Will be list of (j*p+k, value) pairs per term  
    W_list = []  # Will be list of (k*n+i, value) pairs per term
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U_list.append((u_idx, u_val))
        V_list.append((v_idx, v_val))
        W_list.append((w_idx, w_val))
    
    def strassen_2x2x2x2(a_block, b_block, offset_u, offset_v, offset_w):
        """Add 7 Strassen terms for a 2x2 x 2x2 product.
        a_block: list of 4 indices for A block (row-major: a00,a01,a10,a11)
        b_block: list of 4 indices for B block (row-major: b00,b01,b10,b11)
        offset_u: base index for U (i*m+j)
        offset_v: base index for V (j*p+k)
        offset_w: base index for W (k*n+i)
        """
        # a_block = [a00, a01, a10, a11]
        # b_block = [b00, b01, b10, b11]
        # 
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        a00, a01, a10, a11 = a_block
        b00, b01, b10, b11 = b_block
        
        # M1 = (A00+A11)*(B00+B11)
        # U: A00 and A11 have value 1, V: B00 and B11 have value 1, W: C00 and C11
        # Actually, each term is a rank-1: U[i*m+j] * V[j*p+k] * W[k*n+i]
        # For M1: (A00+A11) * (B00+B11) contributes to C00 and C11
        # This is a sum of 4 rank-1 terms, but in the decomposition, we have one term
        # where U has entries at a00 and a11, V has entries at b00 and b11, W has entries at output positions.
        
        # Wait, I need to be more careful. In the tensor decomposition:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (AB)[i,k]
        #
        # For a single Strassen multiplication M1 = (A00+A11)*(B00+B11):
        # This means: U has 1 at positions corresponding to A00 and A11
        # V has 1 at positions corresponding to B00 and B11
        # W has 1 at positions corresponding to C00 and C11 (with appropriate signs)
        
        # The indices:
        # A00: i=0,j=0 -> u_idx = 0*m+0 = 0
        # A01: i=0,j=1 -> u_idx = 0*m+1 = 1
        # A10: i=1,j=0 -> u_idx = 1*m+0 = m
        # A11: i=1,j=1 -> u_idx = 1*m+1 = m+1
        #
        # But these are relative to the block. The actual indices depend on the block position.
        
        # Let me use the offset approach. The a_block indices are the actual (i,j) pairs.
        # So a00 is the (i,j) for A[0,0] of the block, etc.
        
        # For M1: U[a00]=1, U[a11]=1, V[b00]=1, V[b11]=1, W[C00]=1, W[C11]=1
        # But C00 and C11 are output positions.
        
        # Actually, I realize I need to track the output indices too. Let me restructure.
        pass
    
    # This is getting very complex. Let me just try to find a solution via 
    # optimization for a specific rank, or use a known result.
    
    # Actually, let me try a simpler approach: just use the schoolbook but with
    # some clever grouping to get a small improvement.
    
    # Or, let me try to directly construct a rank-78 decomposition using the
    # block structure above.
    
    # Let me define the index mappings:
    # A[i,j] -> U index: i*m + j = i*4 + j
    # B[j,k] -> V index: j*p + k = j*7 + k
    # C[i,k] -> W index: k*n + i = k*3 + i
    
    # Block definitions:
    # A11: i in [0,1], j in [0,1] -> U indices: 0,1,4,5
    # A12: i in [0,1], j in [2,3] -> U indices: 2,3,6,7
    # A21: i=2, j in [0,1] -> U indices: 8,9
    # A22: i=2, j in [2,3] -> U indices: 10,11
    #
    # B11: j in [0,1], k in [0,1] -> V indices: 0,1,7,8
    # B12: j in [0,1], k in [2,3] -> V indices: 2,3,9,10
    # B13: j in [0,1], k in [4,5] -> V indices: 4,5,11,12
    # B14: j in [0,1], k=6 -> V indices: 6,13
    # B21: j in [2,3], k in [0,1] -> V indices: 14,15,21,22
    # B22: j in [2,3], k in [2,3] -> V indices: 16,17,23,24
    # B23: j in [2,3], k in [4,5] -> V indices: 18,19,25,26
    # B24: j in [2,3], k=6 -> V indices: 20,27
    #
    # C11: i in [0,1], k in [0,1] -> W indices: 0,1,3,4
    # C12: i in [0,1], k in [2,3] -> W indices: 6,7,9,10
    # C13: i in [0,1], k in [4,5] -> W indices: 12,13,15,16
    # C14: i in [0,1], k=6 -> W indices: 18,19
    # C21: i=2, k in [0,1] -> W indices: 2,5
    # C22: i=2, k in [2,3] -> W indices: 8,11
    # C23: i=2, k in [4,5] -> W indices: 14,17
    # C24: i=2, k=6 -> W indices: 20
    
    # Now, for each 2x2 x 2x2 product, I'll use Strassen.
    # For C11 = A11*B11 + A12*B21:
    #   Use Strassen for A11*B11 (7 terms) and Strassen for A12*B21 (7 terms)
    #   But the outputs overlap in C11, so I need to be careful.
    
    # Actually, the tensor decomposition allows me to have multiple terms contributing
    # to the same output. So I can just add the Strassen terms for each product.
    
    # Let me define a helper to add a Strassen decomposition for a 2x2 x 2x2 product
    # where the A block is at rows [i1,i2], cols [j1,j2] and B block is at rows [j1,j2], cols [k1,k2]
    # and the output C block is at rows [i1,i2], cols [k1,k2].
    
    def add_strassen_term(U, V, W, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add 7 Strassen terms for 2x2 x 2x2 product.
        a_rows, a_cols: row/col indices of A block (2 each)
        b_rows, b_cols: row/col indices of B block (2 each)  
        c_rows, c_cols: row/col indices of C block (2 each)
        Assumes a_cols == b_rows (the shared dimension).
        """
        # A block indices: (a_rows[0], a_cols[0]), (a_rows[0], a_cols[1]), (a_rows[1], a_cols[0]), (a_rows[1], a_cols[1])
        # B block indices: (b_rows[0], b_cols[0]), (b_rows[0], b_cols[1]), (b_rows[1], b_cols[0]), (b_rows[1], b_cols[1])
        # C block indices: (c_rows[0], c_cols[0]), (c_rows[0], c_cols[1]), (c_rows[1], c_cols[0]), (c_rows[1], c_cols[1])
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        m_val = m  # for U index calculation
        p_val = p  # for V index calculation
        n_val = n  # for W index calculation
        
        def u_idx(i, j):
            return i * m_val + j
        
        def v_idx(j, k):
            return j * p_val + k
        
        def w_idx(i, k):
            return k * n_val + i
        
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00 (+1), C11 (+1)
        U.append({u_idx(*a00): 1, u_idx(*a11): 1})
        V.append({v_idx(*b00): 1, v_idx(*b11): 1})
        W.append({w_idx(*c00): 1, w_idx(*c11): 1})
        
        # M2 = (A10+A11)*B00 -> contributes to C10 (+1)
        U.append({u_idx(*a10): 1, u_idx(*a11): 1})
        V.append({v_idx(*b00): 1})
        W.append({w_idx(*c10): 1})
        
        # M3 = A00*(B01-B11) -> contributes to C01 (+1)
        U.append({u_idx(*a00): 1})
        V.append({v_idx(*b01): 1, v_idx(*b11): -1})
        W.append({w_idx(*c01): 1})
        
        # M4 = A11*(B10-B00) -> contributes to C00 (+1), C10 (+1)
        U.append({u_idx(*a11): 1})
        V.append({v_idx(*b10): 1, v_idx(*b00): -1})
        W.append({w_idx(*c00): 1, w_idx(*c10): 1})
        
        # M5 = (A00+A01)*B11 -> contributes to C00 (-1), C01 (+1)
        U.append({u_idx(*a00): 1, u_idx(*a01): 1})
        V.append({v_idx(*b11): 1})
        W.append({w_idx(*c00): -1, w_idx(*c01): 1})
        
        # M6 = (A10-A00)*(B00+B01) -> contributes to C11 (+1)
        U.append({u_idx(*a10): 1, u_idx(*a00): -1})
        V.append({v_idx(*b00): 1, v_idx(*b01): 1})
        W.append({w_idx(*c11): 1})
        
        # M7 = (A01-A11)*(B10+B11) -> contributes to C00 (+1), C11 (+1)
        U.append({u_idx(*a01): 1, u_idx(*a11): -1})
        V.append({v_idx(*b10): 1, v_idx(*b11): 1})
        W.append({w_idx(*c00): 1, w_idx(*c11): 1})
    
    def add_simple_term(U, V, W, a_idx, b_idx, c_idx, sign=1):
        """Add a single rank-1 term."""
        U.append({a_idx: sign})
        V.append({b_idx: 1})
        W.append({c_idx: 1})
    
    # Initialize lists
    U = []
    V = []
    W = []
    
    # C11 = A11*B11 + A12*B21
    # A11: rows [0,1], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C11: rows [0,1], cols [0,1]
    add_strassen_term(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # A12: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # C11: rows [0,1], cols [0,1]
    add_strassen_term(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # C12 = A11*B12 + A12*B22
    # A11: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # C12: rows [0,1], cols [2,3]
    add_strassen_term(U, V, W, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # A12: rows [0,1], cols [2,3]
    # B22: rows [2,3], cols [2,3]
    # C12: rows [0,1], cols [2,3]
    add_strassen_term(U, V, W, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # C13 = A11*B13 + A12*B23
    # A11: rows [0,1], cols [0,1]
    # B13: rows [0,1], cols [4,5]
    # C13: rows [0,1], cols [4,5]
    add_strassen_term(U, V, W, [0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    
    # A12: rows [0,1], cols [2,3]
    # B23: rows [2,3], cols [4,5]
    # C13: rows [0,1], cols [4,5]
    add_strassen_term(U, V, W, [0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    
    # C14 = A11*B14 + A12*B24
    # A11: rows [0,1], cols [0,1]
    # B14: rows [0,1], col [6]
    # C14: rows [0,1], col [6]
    # This is 2x2 x 2x1, use 4 schoolbook terms
    # A11[0,0]*B14[0,6] -> C14[0,6]
    add_simple_term(U, V, W, 0*4+0, 0*7+6, 6*3+0)
    # A11[0,1]*B14[1,6] -> C14[0,6]
    add_simple_term(U, V, W, 0*4+1, 1*7+6, 6*3+0)
    # A11[1,0]*B14[0,6] -> C14[1,6]
    add_simple_term(U, V, W, 1*4+0, 0*7+6, 6*3+1)
    # A11[1,1]*B14[1,6] -> C14[1,6]
    add_simple_term(U, V, W, 1*4+1, 1*7+6, 6*3+1)
    
    # A12: rows [0,1], cols [2,3]
    # B24: rows [2,3], col [6]
    # C14: rows [0,1], col [6]
    # A12[0,2]*B24[2,6] -> C14[0,6]
    add_simple_term(U, V, W, 0*4+2, 2*7+6, 6*3+0)
    # A12[0,3]*B24[3,6] -> C14[0,6]
    add_simple_term(U, V, W, 0*4+3, 3*7+6, 6*3+0)
    # A12[1,2]*B24[2,6] -> C14[1,6]
    add_simple_term(U, V, W, 1*4+2, 2*7+6, 6*3+1)
    # A12[1,3]*B24[3,6] -> C14[1,6]
    add_simple_term(U, V, W, 1*4+3, 3*7+6, 6*3+1)
    
    # C21 = A21*B11 + A22*B21
    # A21: row [2], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C21: row [2], cols [0,1]
    # 1x2 x 2x2, use 4 schoolbook terms
    # A21[2,0]*B11[0,0] -> C21[2,0]
    add_simple_term(U, V, W, 2*4+0, 0*7+0, 0*3+2)
    # A21[2,0]*B11[0,1] -> C21[2,1]
    add_simple_term(U, V, W, 2*4+0, 0*7+1, 1*3+2)
    # A21[2,1]*B11[1,0] -> C21[2,0]
    add_simple_term(U, V, W, 2*4+1, 1*7+0, 0*3+2)
    # A21[2,1]*B11[1,1] -> C21[2,1]
    add_simple_term(U, V, W, 2*4+1, 1*7+1, 1*3+2)
    
    # A22: row [2], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # C21: row [2], cols [0,1]
    # A22[2,2]*B21[2,0] -> C21[2,0]
    add_simple_term(U, V, W, 2*4+2, 2*7+0, 0*3+2)
    # A22[2,2]*B21[2,1] -> C21[2,1]
    add_simple_term(U, V, W, 2*4+2, 2*7+1, 1*3+2)
    # A22[2,3]*B21[3,0] -> C21[2,0]
    add_simple_term(U, V, W, 2*4+3, 3*7+0, 0*3+2)
    # A22[2,3]*B21[3,1] -> C21[2,1]
    add_simple_term(U, V, W, 2*4+3, 3*7+1, 1*3+2)
    
    # C22 = A21*B12 + A22*B22
    # A21: row [2], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # C22: row [2], cols [2,3]
    add_simple_term(U, V, W, 2*4+0, 0*7+2, 2*3+2)
    add_simple_term(U, V, W, 2*4+0, 0*7+3, 3*3+2)
    add_simple_term(U, V, W, 2*4+1, 1*7+2, 2*3+2)
    add_simple_term(U, V, W, 2*4+1, 1*7+3, 3*3+2)
    
    # A22: row [2], cols [2,3]
    # B22: rows [2,3], cols [2,3]
    # C22: row [2], cols [2,3]
    add_simple_term(U, V, W, 2*4+2, 2*7+2, 2*3+2)
    add_simple_term(U, V, W, 2*4+2, 2*7+3, 3*3+2)
    add_simple_term(U, V, W, 2*4+3, 3*7+2, 2*3+2)
    add_simple_term(U, V, W, 2*4+3, 3*7+3, 3*3+2)
    
    # C23 = A21*B13 + A22*B23
    # A21: row [2], cols [0,1]
    # B13: rows [0,1], cols [4,5]
    # C23: row [2], cols [4,5]
    add_simple_term(U, V, W, 2*4+0, 0*7+4, 4*3+2)
    add_simple_term(U, V, W, 2*4+0, 0*7+5, 5*3+2)
    add_simple_term(U, V, W, 2*4+1, 1*7+4, 4*3+2)
    add_simple_term(U, V, W, 2*4+1, 1*7+5, 5*3+2)
    
    # A22: row [2], cols [2,3]
    # B23: rows [2,3], cols [4,5]
    # C23: row [2], cols [4,5]
    add_simple_term(U, V, W, 2*4+2, 2*7+4, 4*3+2)
    add_simple_term(U, V, W, 2*4+2, 2*7+5, 5*3+2)
    add_simple_term(U, V, W, 2*4+3, 3*7+4, 4*3+2)
    add_simple_term(U, V, W, 2*4+3, 3*7+5, 5*3+2)
    
    # C24 = A21*B14 + A22*B24
    # A21: row [2], cols [0,1]
    # B14: rows [0,1], col [6]
    # C24: row [2], col [6]
    add_simple_term(U, V, W, 2*4+0, 0*7+6, 6*3+2)
    add_simple_term(U, V, W, 2*4+1, 1*7+6, 6*3+2)
    
    # A22: row [2], cols [2,3]
    # B24: rows [2,3], col [6]
    # C24: row [2], col [6]
    add_simple_term(U, V, W, 2*4+2, 2*7+6, 6*3+2)
    add_simple_term(U, V, W, 2*4+3, 3*7+6, 6*3+2)
    
    # Now convert the list of dicts to matrices
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in U[c].items():
            U_mat[idx][c] = val
        for idx, val in V[c].items():
            V_mat[idx][c] = val
        for idx, val in W[c].items():
            W_mat[idx][c] = val
    
    # Verify the decomposition
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            expected = 0.0  # (AB)[i,k] = sum_j A[i,j]*B[j,k]
            # In the tensor, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
            # So (AB)[i,k] = sum_j 1 = m = 4? No, that's not right.
            # The tensor decomposition should satisfy:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
            # But the matrix multiplication is: (AB)[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor T_{ijk} = delta_{i,k} * ... no.
            # 
            # Actually, the matrix multiplication tensor is defined as:
            # T[i*m+j, j*p+k, k*n+i] = 1
            # This means: for each i,j,k, the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c should equal 1.
            # But that's not the matrix multiplication! 
            #
            # Wait, I think I'm confused. Let me re-read the task.
            # "exactly decompose the matrix multiplication tensor"
            # The matrix multiplication tensor for multiplying n x m by m x p has entries:
            # T[i, j, k] where i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
            # T[i,j,k] = 1 if this term contributes to (AB)[i,k]
            # 
            # The flattening is: U index = i*m+j, V index = j*p+k, W index = k*n+i
            # And T[i*m+j, j*p+k, k*n+i] = 1
            #
            # So the condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            #
            # This is NOT the same as (AB)[i,k] = sum_j A[i,j]*B[j,k].
            # The tensor is a 3D array where each entry is 1, and we're decomposing it.
            
            # Let me verify differently: check that for each (i,j,k), the sum equals 1.
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Decomposition is wrong, fall back to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
