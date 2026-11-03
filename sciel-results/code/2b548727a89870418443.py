"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    The key insight: 4x4 * 4x8 can be viewed as two independent 4x4 * 4x4 multiplications
    (splitting the 8 columns into two groups of 4). If we can find a rank-20 decomposition
    for 4x4x4, we get rank 40 for 4x4x8.
    
    However, the best known rank for 4x4x4 is 20 (Strassen-like). Let's use a known
    rank-20 decomposition for 4x4x4.
    
    Actually, let me think more carefully. The standard result is that the tensor rank of
    4x4x4 matrix multiplication is at most 20 (via Strassen's algorithm adapted). 
    The exact minimum is known to be between 18 and 20.
    
    For 4x4x8, we can split into two 4x4x4 blocks, giving rank at most 40.
    
    But we need to construct this explicitly. Let me use a known rank-20 decomposition
    for 4x4x4 based on Strassen's algorithm.
    
    Strassen's 2x2x2 rank-7 algorithm can be extended. For 4x4x4, one approach is to
    use a block structure.
    
    Actually, let me just implement a search for a low-rank decomposition using
    random initialization and gradient-free optimization, or use a known construction.
    
    Let me use a simpler approach: the rank of 4x4x8 is at most 2 * rank(4x4x4).
    If rank(4x4x4) = 20, then rank(4x4x8) <= 40.
    
    I'll construct a rank-20 decomposition for 4x4x4 using a known formula, then
    combine two copies for 4x4x8.
    
    For 4x4x4, one known rank-20 construction uses the following idea:
    Split the 4x4 matrix into 2x2 blocks and apply Strassen recursively.
    
    Let me implement a direct search with random restarts and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # The tensor T[i,j,k] = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Strategy: split the 8 columns into two groups of 4
    # C[:, 0:4] = A @ B[:, 0:4] and C[:, 4:8] = A @ B[:, 4:8]
    # Each is a 4x4 * 4x4 multiplication
    
    # For 4x4x4, use a rank-20 decomposition based on Strassen's algorithm
    
    # Strassen's 2x2x2 rank-7:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    # Each Ci is a 2x2 product, which needs rank 7 via Strassen
    # So total rank = 4 * 7 = 28 for 4x4x4
    
    # That gives rank 28 for 4x4x4, so rank 56 for 4x4x8.
    # Still better than 128!
    
    # Let me implement this Strassen-based construction.
    
    def strassen_2x2(U_2x2, V_2x2, W_2x2, offset):
        """
        Add Strassen's 7-term decomposition for 2x2x2 to the given matrices.
        U_2x2, V_2x2, W_2x2 are the sub-matrices for the 2x2 blocks.
        offset is the column offset.
        """
        for r in range(7):
            if r == 0:
                # M1 = (A00+A11)(B00+B11)
                U_2x2[0][offset + r] += 1  # A00
                U_2x2[3][offset + r] += 1  # A11
                V_2x2[0][offset + r] += 1  # B00
                V_2x2[2][offset + r] += 1  # B11
                W_2x2[0][offset + r] += 1  # C00
                W_2x2[1][offset + r] += 1  # C01
                W_2x2[2][offset + r] += 1  # C10
                W_2x2[3][offset + r] += 1  # C11
            elif r == 1:
                # M2 = (A10+A11)B00
                U_2x2[2][offset + r] += 1  # A10
                U_2x2[3][offset + r] += 1  # A11
                V_2x2[0][offset + r] += 1  # B00
                W_2x2[2][offset + r] += 1  # C10
                W_2x2[3][offset + r] += 1  # C11
            elif r == 2:
                # M3 = A00(B10-B11)
                U_2x2[0][offset + r] += 1  # A00
                V_2x2[1][offset + r] += 1  # B10
                V_2x2[3][offset + r] -= 1  # -B11
                W_2x2[1][offset + r] += 1  # C01
                W_2x2[3][offset + r] += 1  # C11
            elif r == 3:
                # M4 = A11(B01-B10)
                U_2x2[3][offset + r] += 1  # A11
                V_2x2[2][offset + r] += 1  # B01
                V_2x2[1][offset + r] -= 1  # -B10
                W_2x2[0][offset + r] += 1  # C00
                W_2x2[3][offset + r] += 1  # C11
            elif r == 4:
                # M5 = (A00+A01)B11
                U_2x2[0][offset + r] += 1  # A00
                U_2x2[1][offset + r] += 1  # A01
                V_2x2[3][offset + r] += 1  # B11
                W_2x2[0][offset + r] -= 1  # -C00
                W_2x2[1][offset + r] += 1  # C01
            elif r == 5:
                # M6 = (A10-A00)(B00+B10)
                U_2x2[2][offset + r] += 1  # A10
                U_2x2[0][offset + r] -= 1  # -A00
                V_2x2[0][offset + r] += 1  # B00
                V_2x2[1][offset + r] += 1  # B10
                W_2x2[3][offset + r] += 1  # C11
            elif r == 6:
                # M7 = (A01-A11)(B01+B11)
                U_2x2[1][offset + r] += 1  # A01
                U_2x2[3][offset + r] -= 1  # -A11
                V_2x2[2][offset + r] += 1  # B01
                V_2x2[3][offset + r] += 1  # B11
                W_2x2[0][offset + r] += 1  # C00
    
    # For 4x4x4:
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # B is 4x4, split into 2x2 blocks: B00, B01, B10, B11
    # C is 4x4, split into 2x2 blocks: C00, C01, C10, C11
    
    # The indices for 4x4x4:
    # U has 16 rows (4*4), V has 16 rows (4*4), W has 16 rows (4*4)
    # U[i*4+j, r] corresponds to A[i,j]
    # V[j*4+k, r] corresponds to B[j,k]
    # W[k*4+i, r] corresponds to C[i,k]
    
    # Block structure:
    # A00 = rows 0-1, cols 0-1 of A => U indices: (0,0),(0,1),(1,0),(1,1) => U[0:4]
    # A01 = rows 0-1, cols 2-3 of A => U indices: (0,2),(0,3),(1,2),(1,3) => U[2:6] but wait...
    
    # Let me be more careful. U[i*m+j, r] where m=4.
    # A[i,j] is at U[i*4+j, r]
    # A00 block: i in {0,1}, j in {0,1} => U[0:4]
    # A01 block: i in {0,1}, j in {2,3} => U[2:6]... no, U[0*4+2:2*4+2] = U[2:6]? 
    # U[0*4+0]=U[0], U[0*4+1]=U[1], U[0*4+2]=U[2], U[0*4+3]=U[3]
    # U[1*4+0]=U[4], U[1*4+1]=U[5], U[1*4+2]=U[6], U[1*4+3]=U[7]
    # So A00 = U[0:4], A01 = U[2:6]... no that's wrong.
    
    # A[i,j]: row i, col j
    # A00: i=0,1; j=0,1 => indices (0,0),(0,1),(1,0),(1,1) => U[0], U[1], U[4], U[5]
    # A01: i=0,1; j=2,3 => indices (0,2),(0,3),(1,2),(1,3) => U[2], U[3], U[6], U[7]
    # A10: i=2,3; j=0,1 => indices (2,0),(2,1),(3,0),(3,1) => U[8], U[9], U[12], U[13]
    # A11: i=2,3; j=2,3 => indices (2,2),(2,3),(3,2),(3,3) => U[10], U[11], U[14], U[15]
    
    # Similarly for V[j,k]:
    # B00: j=0,1; k=0,1 => V[0], V[1], V[4], V[5]
    # B01: j=0,1; k=2,3 => V[2], V[3], V[6], V[7]
    # B10: j=2,3; k=0,1 => V[8], V[9], V[12], V[13]
    # B11: j=2,3; k=2,3 => V[10], V[11], V[14], V[15]
    
    # For W[k,i] (note: W[k*n+i, r] where n=4):
    # C00: i=0,1; k=0,1 => W[0*4+0]=W[0], W[0*4+1]=W[1], W[1*4+0]=W[4], W[1*4+1]=W[5]
    # C01: i=0,1; k=2,3 => W[2], W[3], W[6], W[7]
    # C10: i=2,3; k=0,1 => W[8], W[9], W[12], W[13]
    # C11: i=2,3; k=2,3 => W[10], W[11], W[14], W[15]
    
    # Now, for the 4x4x4 decomposition, we apply Strassen to each of the 4 output blocks.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci is a 2x2 product of two 2x2 matrices. We can use Strassen's rank-7 for each.
    # But we need to be careful: Strassen's algorithm computes the product of two 2x2 matrices.
    # For C00 = A00*B00 + A01*B10, this is a sum of two 2x2 products.
    # We can compute A00*B00 with rank 7 and A01*B10 with rank 7, total rank 14 for C00.
    # Similarly for each Ci, total rank = 4 * 14 = 56 for 4x4x4.
    
    # That gives rank 56 for 4x4x4, so rank 112 for 4x4x8. Still better than 128!
    
    # Let me implement this.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112 (m*p = 4*8 = 32)
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112 (p*n = 8*4 = 32)
    
    # For the 4x4x8 case:
    # A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A @ B_left, C_right = A @ B_right
    # Each is a 4x4x4 multiplication
    
    # For C_left (columns 0-3 of C):
    # U indices: A[i,j] => U[i*4+j, r]
    # V indices: B[j,k] where k in 0-3 => V[j*8+k, r]
    # W indices: C[i,k] where k in 0-3 => W[k*4+i, r]
    
    # For C_right (columns 4-7 of C):
    # U indices: A[i,j] => U[i*4+j, r]
    # V indices: B[j,k] where k in 4-7 => V[j*8+k, r]
    # W indices: C[i,k] where k in 4-7 => W[k*4+i, r]
    
    def add_strassen_block(U, V, W, A_indices, B_indices, C_indices, col_offset):
        """
        Add Strassen-based decomposition for a 4x4x4 block.
        A_indices: list of 4 lists, each with 4 U row indices for A00, A01, A10, A11
        B_indices: list of 4 lists, each with 4 V row indices for B00, B01, B10, B11
        C_indices: list of 4 lists, each with 4 W row indices for C00, C01, C10, C11
        col_offset: offset for the rank columns
        """
        # For each output block Ci (i in 0..3), compute Ci = Ai1*Bj1 + Ai2*Bj2
        # Ci0 = A00*B00 + A01*B10
        # Ci1 = A00*B01 + A01*B11
        # Ci2 = A10*B00 + A11*B10
        # Ci3 = A10*B01 + A11*B11
        
        # For each product, use Strassen's rank-7
        products = [
            (0, 0, 0, 1),  # C00 = A00*B00 + A01*B10
            (0, 1, 0, 1),  # C01 = A00*B01 + A01*B11
            (2, 0, 2, 1),  # C10 = A10*B00 + A11*B10
            (2, 1, 2, 1),  # C11 = A10*B01 + A11*B11
        ]
        
        # Actually, let me just do it directly: for each of the 4 output blocks,
        # we have two 2x2 products. Use Strassen for each.
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # For A00*B00: use Strassen with A=A00, B=B00, output to C00
        # For A01*B10: use Strassen with A=A01, B=B10, output to C00 (add)
        # etc.
        
        # Let me define a helper that adds Strassen's 7 terms for a single 2x2x2 product
        def add_strassen_2x2(a_idx, b_idx, c_idx, off):
            """
            a_idx: 4 indices for A (A00,A01,A10,A11) in the 2x2 block
            b_idx: 4 indices for B (B00,B01,B10,B11) in the 2x2 block
            c_idx: 4 indices for C (C00,C01,C10,C11) in the 2x2 block
            off: column offset
            """
            # Strassen's 7 products:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B10)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            terms = [
                # (a_coeff, b_coeff, c_coeff) for each of the 4 positions
                # M1: A00+A11 times B00+B11
                [(1, 1, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0)],  # A: A00=1,A11=1; B: B00=1,B11=1; C: C00=1,C01=1,C10=1,C11=1
                # Wait, I need to be more careful.
            ]
            
            # Let me just hardcode the 7 terms.
            # Each term has: a_coeffs (4 values for A00,A01,A10,A11), 
            #                b_coeffs (4 values for B00,B01,B10,B11),
            #                c_coeffs (4 values for C00,C01,C10,C11)
            
            terms = [
                # M1 = (A00+A11)(B00+B11) => C00+=1, C01+=1, C10+=1, C11+=1
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
                # M2 = (A10+A11)B00 => C10+=1, C11+=1
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
                # M3 = A00(B10-B11) => C01+=1, C11+=1
                ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
                # M4 = A11(B01-B10) => C00+=1, C11+=1
                ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]),
                # M5 = (A00+A01)B11 => C00-=1, C01+=1
                ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                # M6 = (A10-A00)(B00+B10) => C11+=1
                ([ -1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
                # M7 = (A01-A11)(B01+B11) => C00+=1
                ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
            ]
            
            for t in range(7):
                col = off + t
                a_c, b_c, c_c = terms[t]
                for pos in range(4):
                    if a_c[pos] != 0:
                        U[a_idx[pos]][col] += a_c[pos]
                    if b_c[pos] != 0:
                        V[b_idx[pos]][col] += b_c[pos]
                    if c_c[pos] != 0:
                        W[c_idx[pos]][col] += c_c[pos]
        
        # Now define the index sets for the 2x2 blocks
        # A00: [A_idx[0], A_idx[1], A_idx[4], A_idx[5]]  (rows 0,1; cols 0,1)
        # A01: [A_idx[2], A_idx[3], A_idx[6], A_idx[7]]  (rows 0,1; cols 2,3)
        # A10: [A_idx[8], A_idx[9], A_idx[12], A_idx[13]]  (rows 2,3; cols 0,1)
        # A11: [A_idx[10], A_idx[11], A_idx[14], A_idx[15]]  (rows 2,3; cols 2,3)
        
        # B00: [B_idx[0], B_idx[1], B_idx[8], B_idx[9]]  (rows 0,1; cols 0,1)
        # B01: [B_idx[2], B_idx[3], B_idx[10], B_idx[11]]  (rows 0,1; cols 2,3)
        # B10: [B_idx[4], B_idx[5], B_idx[12], B_idx[13]]  (rows 2,3; cols 0,1)
        # B11: [B_idx[6], B_idx[7], B_idx[14], B_idx[15]]  (rows 2,3; cols 2,3)
        
        # C00: [C_idx[0], C_idx[1], C_idx[4], C_idx[5]]  (rows 0,1; cols 0,1)
        # C01: [C_idx[2], C_idx[3], C_idx[6], C_idx[7]]  (rows 0,1; cols 2,3)
        # C10: [C_idx[8], C_idx[9], C_idx[12], C_idx[13]]  (rows 2,3; cols 0,1)
        # C11: [C_idx[10], C_idx[11], C_idx[14], C_idx[15]]  (rows 2,3; cols 2,3)
        
        # Wait, I need to be careful about the ordering. The 2x2 block A00 has entries:
        # A[0,0], A[0,1], A[1,0], A[1,1]
        # In the U matrix, these are at U[0*4+0]=U[0], U[0*4+1]=U[1], U[1*4+0]=U[4], U[1*4+1]=U[5]
        # So A00 indices: [0, 1, 4, 5]
        # A01: A[0,2], A[0,3], A[1,2], A[1,3] => U[2], U[3], U[6], U[7]
        # A10: A[2,0], A[2,1], A[3,0], A[3,1] => U[8], U[9], U[12], U[13]
        # A11: A[2,2], A[2,3], A[3,2], A[3,3] => U[10], U[11], U[14], U[15]
        
        # For B (4x4 in the 4x4x4 case), V[j*4+k]:
        # B00: B[0,0], B[0,1], B[1,0], B[1,1] => V[0], V[1], V[4], V[5]
        # B01: B[0,2], B[0,3], B[1,2], B[1,3] => V[2], V[3], V[6], V[7]
        # B10: B[2,0], B[2,1], B[3,0], B[3,1] => V[8], V[9], V[12], V[13]
        # B11: B[2,2], B[2,3], B[3,2], B[3,3] => V[10], V[11], V[14], V[15]
        
        # For C (4x4 in the 4x4x4 case), W[k*4+i]:
        # C00: C[0,0], C[0,1], C[1,0], C[1,1] => W[0], W[1], W[4], W[5]
        # C01: C[0,2], C[0,3], C[1,2], C[1,3] => W[2], W[3], W[6], W[7]
        # C10: C[2,0], C[2,1], C[3,0], C[3,1] => W[8], W[9], W[12], W[13]
        # C11: C[2,2], C[2,3], C[3,2], C[3,3] => W[10], W[11], W[14], W[15]
        
        # Now, for the 4x4x8 case, the V and W matrices have different dimensions.
        # V is (m*p, rank) = (4*8, rank) = (32, rank)
        # W is (p*n, rank) = (8*4, rank) = (32, rank)
        
        # For the left half (columns 0-3):
        # B_left[j,k] where k in 0-3 => V[j*8+k, r]
        # C_left[i,k] where k in 0-3 => W[k*4+i, r]
        
        # B00_left: B[0,0], B[0,1], B[1,0], B[1,1] => V[0], V[1], V[8], V[9]
        # B01_left: B[0,2], B[0,3], B[1,2], B[1,3] => V[2], V[3], V[10], V[11]
        # B10_left: B[2,0], B[2,1], B[3,0], B[3,1] => V[16], V[17], V[24], V[25]
        # B11_left: B[2,2], B[2,3], B[3,2], B[3,3] => V[18], V[19], V[26], V[27]
        
        # C00_left: C[0,0], C[0,1], C[1,0], C[1,1] => W[0], W[1], W[4], W[5]
        # C01_left: C[0,2], C[0,3], C[1,2], C[1,3] => W[2], W[3], W[6], W[7]
        # C10_left: C[2,0], C[2,1], C[3,0], C[3,1] => W[8], W[9], W[12], W[13]
        # C11_left: C[2,2], C[2,3], C[3,2], C[3,3] => W[10], W[11], W[14], W[15]
        
        # For the right half (columns 4-7):
        # B_right[j,k] where k in 4-7 => V[j*8+k, r]
        # C_right[i,k] where k in 4-7 => W[k*4+i, r]
        
        # B00_right: B[0,4], B[0,5], B[1,4], B[1,5] => V[4], V[5], V[12], V[13]
        # B01_right: B[0,6], B[0,7], B[1,6], B[1,7] => V[6], V[7], V[14], V[15]
        # B10_right: B[2,4], B[2,5], B[3,4], B[3,5] => V[20], V[21], V[28], V[29]
        # B11_right: B[2,6], B[2,7], B[3,6], B[3,7] => V[22], V[23], V[30], V[31]
        
        # C00_right: C[0,4], C[0,5], C[1,4], C[1,5] => W[16], W[17], W[20], W[21]
        # C01_right: C[0,6], C[0,7], C[1,6], C[1,7] => W[18], W[19], W[22], W[23]
        # C10_right: C[2,4], C[2,5], C[3,4], C[3,5] => W[24], W[25], W[28], W[29]
        # C11_right: C[2,6], C[2,7], C[3,6], C[3,7] => W[26], W[27], W[30], W[31]
        
        # A indices are the same for both halves:
        # A00: [0, 1, 4, 5]
        # A01: [2, 3, 6, 7]
        # A10: [8, 9, 12, 13]
        # A11: [10, 11, 14, 15]
        
        A_blocks = {
            0: [0, 1, 4, 5],   # A00
            1: [2, 3, 6, 7],   # A01
            2: [8, 9, 12, 13], # A10
            3: [10, 11, 14, 15] # A11
        }
        
        # Left half
        B_blocks_left = {
            0: [0, 1, 8, 9],    # B00
            1: [2, 3, 10, 11],  # B01
            2: [16, 17, 24, 25], # B10
            3: [18, 19, 26, 27]  # B11
        }
        C_blocks_left = {
            0: [0, 1, 4, 5],    # C00
            1: [2, 3, 6, 7],    # C01
            2: [8, 9, 12, 13],  # C10
            3: [10, 11, 14, 15] # C11
        }
        
        # Right half
        B_blocks_right = {
            0: [4, 5, 12, 13],    # B00
            1: [6, 7, 14, 15],    # B01
            2: [20, 21, 28, 29],  # B10
            3: [22, 23, 30, 31]   # B11
        }
        C_blocks_right = {
            0: [16, 17, 20, 21],    # C00
            1: [18, 19, 22, 23],    # C01
            2: [24, 25, 28, 29],    # C10
            3: [26, 27, 30, 31]     # C11
        }
        
        # Now add the Strassen decompositions
        # For each half, we have 4 output blocks, each requiring 2 Strassen decompositions (for the two products)
        # Total: 2 halves * 4 blocks * 2 products * 7 terms = 112 terms
        
        # Left half
        col_off = 0
        # C00_left = A00*B00 + A01*B10
        add_strassen_2x2(A_blocks[0], B_blocks_left[0], C_blocks_left[0], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[1], B_blocks_left[2], C_blocks_left[0], col_off)
        col_off += 7
        # C01_left = A00*B01 + A01*B11
        add_strassen_2x2(A_blocks[0], B_blocks_left[1], C_blocks_left[1], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[1], B_blocks_left[3], C_blocks_left[1], col_off)
        col_off += 7
        # C10_left = A10*B00 + A11*B10
        add_strassen_2x2(A_blocks[2], B_blocks_left[0], C_blocks_left[2], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[3], B_blocks_left[2], C_blocks_left[2], col_off)
        col_off += 7
        # C11_left = A10*B01 + A11*B11
        add_strassen_2x2(A_blocks[2], B_blocks_left[1], C_blocks_left[3], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[3], B_blocks_left[3], C_blocks_left[3], col_off)
        col_off += 7
        
        # Right half
        # C00_right = A00*B00_right + A01*B10_right
        add_strassen_2x2(A_blocks[0], B_blocks_right[0], C_blocks_right[0], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[1], B_blocks_right[2], C_blocks_right[0], col_off)
        col_off += 7
        # C01_right = A00*B01_right + A01*B11_right
        add_strassen_2x2(A_blocks[0], B_blocks_right[1], C_blocks_right[1], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[1], B_blocks_right[3], C_blocks_right[1], col_off)
        col_off += 7
        # C10_right = A10*B00_right + A11*B10_right
        add_strassen_2x2(A_blocks[2], B_blocks_right[0], C_blocks_right[2], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[3], B_blocks_right[2], C_blocks_right[2], col_off)
        col_off += 7
        # C11_right = A10*B01_right + A11*B11_right
        add_strassen_2x2(A_blocks[2], B_blocks_right[1], C_blocks_right[3], col_off)
        col_off += 7
        add_strassen_2x2(A_blocks[3], B_blocks_right[3], C_blocks_right[3], col_off)
        col_off += 7
        
        # Verify that col_off == rank
        assert col_off == rank, f"col_off={col_off}, rank={rank}"
    
    add_strassen_block(U, V, W, None, None, None, 0)
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta(i,j,k) for the multiplication tensor
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the tensor is T[a,b,c] where a indexes A, b indexes B, c indexes C
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0
    
    # Let's verify by checking that the reconstructed tensor matches
    # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For efficiency, let's just check a few random instances
    random.seed(42)
    for trial in range(10):
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 8)
        C_test = A_test @ B_test
        
        # Compute using the decomposition
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] * A[i,j] * B[j,k]... no
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # And we claim: sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] = delta for the right combination
        
        # Let me just verify the tensor directly
        # The tensor T has T[(i,j),(j,k),(i,k)] = 1
        # So sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] should equal 1 if the indices are consistent
        
        # Actually, the correct check is:
        # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Wait, that's not right either. The tensor is:
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # where T_{(i,j),(j,k),(i,k)} = 1 and T is 0 otherwise.
        
        # Let me just check a few specific entries
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    expected = 1.0
                    actual = sum(U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*4+i, r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        print(f"Mismatch at i={i}, j={j}, k={k}: expected={expected}, actual={actual}")
                        return {"U": [[1]*128 + [0]*0 for _ in range(16)], 
                                "V": [[1]*128 + [0]*0 for _ in range(32)], 
                                "W": [[1]*128 + [0]*0 for _ in range(32)]}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
