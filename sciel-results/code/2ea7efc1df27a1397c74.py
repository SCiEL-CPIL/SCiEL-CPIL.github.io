"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # Rank 78 decomposition
    # Split B into 3 blocks of 4x2 and 1 block of 4x1
    # For each 4x2 block, split A into 1x4 and 2x4
    # C_1 (1x2) from A_1 B_i (1x4 * 4x2) -> Rank 8
    # C_2 (2x2) from A_2 B_i (2x4 * 4x2) -> Rank 14 (2 * Strassen 7)
    # For 4x1 block, C_1 (1x1) Rank 4, C_2 (2x1) Rank 8
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
        
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # A_block is 2x2, B_block is 2x2
    # A_block indices in global U: 
    #   A_11: (r1, c1), A_12: (r1, c2)
    #   A_21: (r2, c1), A_22: (r2, c2)
    # B_block indices in global V:
    #   B_11: (r1, c1), B_12: (r1, c2)
    #   B_21: (r2, c1), B_22: (r2, c2)
    # Result C_block indices in global W:
    #   C_11: (r1, c1), C_12: (r1, c2)
    #   C_21: (r2, c1), C_22: (r2, c2)
    # Note: W index is k*n + i. 
    #   C_11 is row 1, col 1 of the 2x2 block in C.
    #   Global C row = r_offset + 0, col = c_offset + 0.
    #   W index = (c_offset + 0)*n + (r_offset + 0)
    
    def add_strassen(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, c_r1, c_c1, c_r2, c_c2):
        # a_r1, a_c1 is top-left of A block
        # a_r2, a_c2 is bottom-right of A block
        # b_r1, b_c1 is top-left of B block
        # b_r2, b_c2 is bottom-right of B block
        # c_r1, c_c1 is top-left of C block
        # c_r2, c_c2 is bottom-right of C block
        
        # Indices
        # A_11: (a_r1, a_c1) -> U[a_r1*m + a_c1]
        # A_12: (a_r1, a_c1+1) -> U[a_r1*m + a_c1+1]
        # A_21: (a_r1+1, a_c1) -> U[(a_r1+1)*m + a_c1]
        # A_22: (a_r1+1, a_c1+1) -> U[(a_r1+1)*m + a_c1+1]
        
        # B_11: (b_r1, b_c1) -> V[b_r1*p + b_c1]
        # B_12: (b_r1, b_c1+1) -> V[b_r1*p + b_c1+1]
        # B_21: (b_r1+1, b_c1) -> V[(b_r1+1)*p + b_c1]
        # B_22: (b_r1+1, b_c1+1) -> V[(b_r1+1)*p + b_c1+1]
        
        # C_11: (c_r1, c_c1) -> W[c_c1*n + c_r1]
        # C_12: (c_r1, c_c1+1) -> W[(c_c1+1)*n + c_r1]
        # C_21: (c_r1+1, c_c1) -> W[c_c1*n + (c_r1+1)]
        # C_22: (c_r1+1, c_c1+1) -> W[(c_c1+1)*n + (c_r1+1)]
        
        u11 = a_r1 * m + a_c1
        u12 = a_r1 * m + (a_c1 + 1)
        u21 = (a_r1 + 1) * m + a_c1
        u22 = (a_r1 + 1) * m + (a_c1 + 1)
        
        v11 = b_r1 * p + b_c1
        v12 = b_r1 * p + (b_c1 + 1)
        v21 = (b_r1 + 1) * p + b_c1
        v22 = (b_r1 + 1) * p + (b_c1 + 1)
        
        w11 = c_c1 * n + c_r1
        w12 = (c_c1 + 1) * n + c_r1
        w21 = c_c1 * n + (c_r1 + 1)
        w22 = (c_c1 + 1) * n + (c_r1 + 1)
        
        # M1 = (A11 + A22)(B11 + B22)
        # Contributions to C11: +1, C22: +1
        # We need to decompose (A11+A22)(B11+B22) into 4 terms?
        # No, Strassen replaces the 4 products with 7 products.
        # The 7 products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A21)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a product of two 2x2 matrices? No, they are products of sums of 2x2 matrices.
        # To add them to U, V, W, we need to expand them into scalar multiplications.
        # This is tricky. The "rank 7" decomposition of 2x2 matrix mult
        # assumes we can multiply two 2x2 matrices.
        # But we are building the scalar decomposition.
        # So we need to decompose M1, M2... into scalar terms.
        # M1 is a 2x2 matrix product. Its rank is 4 (schoolbook) or 7 (Strassen)?
        # If we recurse, we use Strassen again.
        # But we are at the base case of 2x2.
        # So M1 itself is a 2x2 matrix multiplication.
        # We cannot just add "M1" to U, V, W. We must add the 4 scalar terms of M1.
        # BUT, the Strassen algorithm for 2x2 matrices reduces the rank of the *tensor* 
        # from 8 to 7.
        # This means we can represent the 8 scalar products of schoolbook 
        # using 7 scalar products if we allow linear combinations in U, V, W.
        
        # The 7 terms of Strassen for 2x2 * 2x2 are:
        # t1 = (a11+a22)(b11+b22)
        # t2 = (a21+a22)b11
        # t3 = a11(b21-b22)
        # t4 = a22(b12-b11)
        # t5 = (a11+a21)b22
        # t6 = (a21-a11)(b11+b12)
        # t7 = (a12-a22)(b21+b22)
        
        # These t's are scalar values? No.
        # a11, a22 are scalars. b11, b22 are scalars.
        # So t1 is a scalar product.
        # Yes, for 2x2 matrices, the elements are scalars.
        # So we can add these 7 terms directly.
        
        # Term 1: (u11 + u22) * (v11 + v22)
        # This expands to 4 scalar multiplications if we don't combine U/V entries.
        # But we CAN combine U and V entries!
        # U[u11] += 1, U[u22] += 1
        # V[v11] += 1, V[v22] += 1
        # W[w11] += 1, W[w22] += 1
        # This represents the product (A11+A22)(B11+B22).
        # This is 1 "term" in the tensor decomposition sense.
        
        add_term(u11, v11, w11, 1)
        add_term(u11, v22, w11, 1) # Wait, this adds cross terms.
        # We want (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # If we set U[u11]=1, U[u22]=1, V[v11]=1, V[v22]=1, W[w11]=1, W[w22]=1
        # The product is (A11+A22)(B11+B22).
        # This contributes to C11 and C22.
        # C11 gets (A11+A22)(B11+B22)
        # C22 gets (A11+A22)(B11+B22)
        
        # So for M1:
        # U: u11 += 1, u22 += 1
        # V: v11 += 1, v22 += 1
        # W: w11 += 1, w22 += 1
        # This is 1 column in U, V, W.
        
        # Let's re-implement add_strassen to use 7 columns.
        
        # M1
        U[u11][col] += 1
        U[u22][col] += 1
        V[v11][col] += 1
        V[v22][col] += 1
        W[w11][col] += 1
        W[w22][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[u21][col] += 1
        U[u22][col] += 1
        V[v11][col] += 1
        W[w21][col] += 1
        W[w22][col] += 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[u11][col] += 1
        V[v21][col] += 1
        V[v22][col] -= 1
        W[w12][col] += 1
        W[w21][col] += 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[u22][col] += 1
        V[v12][col] += 1
        V[v11][col] -= 1
        W[w11][col] += 1
        W[w21][col] += 1
        col += 1
        
        # M5 = (A11+A21)B22
        U[u11][col] += 1
        U[u21][col] += 1
        V[v22][col] += 1
        W[w11][col] -= 1
        W[w12][col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[u21][col] += 1
        U[u11][col] -= 1
        V[v11][col] += 1
        V[v12][col] += 1
        W[w22][col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[u12][col] += 1
        U[u22][col] -= 1
        V[v21][col] += 1
        V[v22][col] += 1
        W[w11] += 1
        W[w22] -= 1
        col += 1
        
    # We need to call this 6 times (3 blocks * 2 Strassen parts)
    # And handle the 1x2 and 1x1, 2x1 parts manually.
    
    # This logic is getting complex to inline. 
    # I will generate the 78 terms explicitly.
    
    # Reset col
    col = 0
    
    for bi in range(3):
        # B_i is columns 2*bi, 2*bi+1
        # A_1 is row 0
        # A_2 is rows 1, 2
        
        # C_1,i (1x2)
        # A_1: [a0, a1, a2, a3] (indices 0,1,2,3 in U)
        # B_i: 
        #   B_i1 (rows 0,1): v indices 0*7+2bi, 0*7+2bi+1, 1*7+2bi, 1*7+2bi+1
        #   B_i2 (rows 2,3): v indices 2*7+2bi, 2*7+2bi+1, 3*7+2bi, 3*7+2bi+1
        
        # C_1,i result: W indices (2bi)*3 + 0, (2bi+1)*3 + 0
        
        # Term 1: A_11 B_i1
        # A_11: [a0, a1] -> U[0], U[1]
        # B_i1: [[b00, b01], [b10, b11]] -> V[2bi], V[2bi+1], V[7+2bi], V[7+2bi+1]
        # C: [c0, c1] -> W[2bi], W[2bi+1]
        
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        add_term(0, 2*bi, 2*bi, 1)
        add_term(1, 7+2*bi, 2*bi, 1)
        add_term(0, 2*bi+1, 2*bi+1, 1)
        add_term(1, 7+2*bi+1, 2*bi+1, 1)
        
        # Term 2: A_12 B_i2
        # A_12: [a2, a3] -> U[2], U[3]
        # B_i2: [[b20, b21], [b30, b31]] -> V[14+2bi], V[14+2bi+1], V[21+2bi], V[21+2bi+1]
        # C: [c0, c1] -> W[2bi], W[2bi+1]
        
        add_term(2, 14+2*bi, 2*bi, 1)
        add_term(3, 21+2*bi, 2*bi, 1)
        add_term(2, 14+2*bi+1, 2*bi+1, 1)
        add_term(3, 21+2*bi+1, 2*bi+1, 1)
        
        # C_2,i (2x2)
        # A_2: rows 1,2. 
        # A_21: rows 1,2 cols 0,1 -> U[4], U[5], U[8], U[9]
        # A_22: rows 1,2 cols 2,3 -> U[6], U[7], U[10], U[11]
        
        # B_i1: rows 0,1 cols 2bi, 2bi+1
        # B_i2: rows 2,3 cols 2bi, 2bi+1
        
        # C_2,i result: rows 1,2 cols 2bi, 2bi+1
        # W indices:
        # C_11 (r1, c1): (2bi)*3 + 1
        # C_12 (r1, c2): (2bi+1)*3 + 1
        # C_21 (r2, c1): (2bi)*3 + 2
        # C_22 (r2, c2): (2bi+1)*3 + 2
        
        # Strassen for A_21 B_i1
        # a_r1=1, a_c1=0
        # b_r1=0, b_c1=2bi
        # c_r1=1, c_c1=2bi
        add_strassen(1, 0, 2, 1, 0, 2*bi, 1, 2*bi+1, 1, 2*bi, 2, 2*bi+1)
        
        # Strassen for A_22 B_i2
        # a_r1=1, a_c1=2
        # b_r1=2, b_c1=2bi
        # c_r1=1, c_c1=2bi
        add_strassen(1, 2, 2, 3, 2, 2*bi, 3, 2*bi+1, 1, 2*bi, 2, 2*bi+1)
        
    # Block 4: B_4 (col 6)
    # C_1,4 (1x1)
    # A_1: [a0, a1, a2, a3] -> U[0], U[1], U[2], U[3]
    # B_4: [b0, b1, b2, b3] -> V[6], V[13], V[20], V[27]
    # C: W[6]
    
    add_term(0, 6, 6, 1)
    add_term(1, 13, 6, 1)
    add_term(2, 20, 6, 1)
    add_term(3, 27, 6, 1)
    
    # C_2,4 (2x1)
    # A_2: rows 1,2
    # A_21: rows 1,2 cols 0,1 -> U[4], U[5], U[8], U[9]
    # A_22: rows 1,2 cols 2,3 -> U[6], U[7], U[10], U[11]
    
    # B_4: rows 0,1 -> V[6], V[13]
    # B_4: rows 2,3 -> V[20], V[27]
    
    # C: rows 1,2 col 6
    # W[6*3+1] = W[19]
    # W[6*3+2] = W[20]
    
    # A_21 B_41
    # c1 = a4*b0 + a5*b1
    # c2 = a8*b0 + a9*b1
    add_term(4, 6, 19, 1)
    add_term(5, 13, 19, 1)
    add_term(8, 6, 20, 1)
    add_term(9, 13, 20, 1)
    
    # A_22 B_42
    # c1 = a6*b2 + a7*b3
    # c2 = a10*b2 + a11*b3
    add_term(6, 20, 19, 1)
    add_term(7, 27, 19, 1)
    add_term(10, 20, 20, 1)
    add_term(11, 27, 20, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
