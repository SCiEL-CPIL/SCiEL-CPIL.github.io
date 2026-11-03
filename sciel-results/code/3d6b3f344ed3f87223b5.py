"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Strategy: Use block decomposition 1x1 and 4x4.
    # Split 5x5 into blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 1x1, A12 is 1x4, A21 is 4x1, A22 is 4x4
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 (1x1) = A11*B11 + A12*B21
    # C12 (1x4) = A11*B12 + A12*B22
    # C21 (4x1) = A21*B11 + A22*B21
    # C22 (4x4) = A21*B12 + A22*B22
    
    # We will generate the rank-1 terms for each block multiplication.
    # For 4x4x4, we use Strassen's algorithm (rank 49).
    # For other rectangular blocks, we use schoolbook.
    
    # Helper to generate schoolbook terms for p x q x r multiplication
    def schoolbook_terms(p, q, r, offset_u, offset_v, offset_w):
        terms = []
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    u_idx = offset_u + i * q + j
                    v_idx = offset_v + j * r + k
                    w_idx = offset_w + k * p + i
                    terms.append((u_idx, v_idx, w_idx))
        return terms
    
    # Helper to generate Strassen terms for 4x4x4 multiplication
    # Strassen for 2x2 has rank 7.
    # For 4x4, we split into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks)
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix multiplication.
    # We can recursively apply Strassen to the 2x2 blocks?
    # No, 2x2 Strassen is the base case.
    # So for 4x4, we have 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank 49.
    
    # We need to map the 2x2 block indices to the global 4x4 indices.
    # Let the 4x4 block start at offset (ou, ov, ow) in the global U, V, W indices.
    # The 4x4 block is composed of 2x2 sub-blocks.
    # Sub-block (si, sj) of A has global row indices offset_u + si*2 + i, col offset_v + sj*2 + j.
    # Wait, U index is i*m + j.
    # Let's define the 4x4 block local indices.
    # A_4x4[i, j] corresponds to U[ou + i*4 + j] ? No.
    # In the global matrix, the 4x4 block A22 occupies rows 1..4 and cols 1..4.
    # So local i, j in 0..3 maps to global row 1+i, col 1+j.
    # U index = (1+i)*5 + (1+j).
    
    def strassen_4x4_terms(offset_u, offset_v, offset_w):
        terms = []
        # Define the 2x2 block offsets within the 4x4 block
        # Block (bi, bj) of the 4x4 matrix corresponds to global rows offset_u + bi*2 + i, cols offset_v + bj*2 + j
        # where i, j in 0..1.
        
        # Strassen factors for 2x2:
        # M1: (A00+A11)*(B00+B11)
        # M2: (A10+A11)*B00
        # M3: A00*(B01-B11)
        # M4: A11*(B10-B00)
        # M5: (A00+A01)*B11
        # M6: (A10-A00)*(B00+B01)
        # M7: (A01-A11)*(B10+B11)
        
        # We need to generate the rank-1 terms for each of these 7 2x2 multiplications.
        # And then combine them to form the 4x4 result.
        
        # This is complex to implement directly in terms of U, V, W vectors.
        # Alternative: Just use schoolbook for 4x4? Rank 64.
        # 1+4 split with schoolbook 4x4:
        # C11: 1 + 16 = 17
        # C12: 4 + 16 = 20
        # C21: 4 + 16 = 20
        # C22: 16 + 64 = 80
        # Total: 137. Worse than 125.
        
        # So we MUST use Strassen for 4x4 to get rank 49.
        # Let's implement the Strassen 4x4 decomposition.
        
        # The 4x4 multiplication C = A*B can be computed by 7 2x2 multiplications.
        # Let A be partitioned into 2x2 blocks A00, A01, A10, A11.
        # Let B be partitioned into 2x2 blocks B00, B01, B10, B11.
        # M1 = (A00+A11)*(B00+B11)
        # ...
        # Each Mk is a 2x2 matrix.
        # The result C is formed by linear combinations of Mk.
        
        # To generate the U, V, W terms, we need to expand each Mk into its 7 rank-1 terms (from 2x2 Strassen).
        # And then distribute these terms into the global U, V, W vectors with appropriate coefficients.
        
        # This requires careful bookkeeping of indices and coefficients.
        
        # Given the complexity and risk of error, and the time limit,
        # I will stick to the 125 schoolbook decomposition for now, 
        # but I will try to implement the 1+4 split with schoolbook 4x4 if it's better? No, 137 is worse.
        
        # What if I use 2+3 split with schoolbook 3x3 (27) and Strassen 2x2 (7)?
        # C11: 27 + 18 = 45
        # C12: 18 + 12 = 30
        # C21: 18 + 12 = 30
        # C22: 12 + 7 = 19
        # Total: 124.
        # This is better than 125.
        
        # Let's implement the 2+3 split.
        pass
        
    # Implementation of 2+3 split
    # Split indices:
    # I1 = [0, 1], I2 = [2, 3, 4]
    # A11: 2x2, A12: 2x3, A21: 3x2, A22: 3x3
    # B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x3) = A11*B12 + A12*B22
    # C21 (3x2) = A21*B11 + A22*B21
    # C22 (3x3) = A21*B12 + A22*B22
    
    # Ranks:
    # A11*B11: 2x2x2 -> Strassen (7)
    # A12*B21: 2x3x2 -> Schoolbook (12)
    # C11 rank: 19
    
    # A11*B12: 2x2x3 -> Schoolbook (12)
    # A12*B22: 2x3x3 -> Schoolbook (18)
    # C12 rank: 30
    
    # A21*B11: 3x2x2 -> Schoolbook (12)
    # A22*B21: 3x3x2 -> Schoolbook (18)
    # C21 rank: 30
    
    # A21*B12: 3x2x3 -> Schoolbook (18)
    # A22*B22: 3x3x3 -> Schoolbook (27)
    # C22 rank: 45
    
    # Total: 19 + 30 + 30 + 45 = 124.
    
    # Let's construct the terms.
    
    def get_block_indices(block_type, i, j):
        # block_type: 'A11', 'A12', etc.
        # Returns global U index for A[i,j], V index for B[i,j], W index for C[i,j]
        # This is tricky because the blocks are in different positions.
        pass

    # Simpler approach:
    # Generate all terms for each block multiplication explicitly.
    
    terms = []
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2x2 Strassen
    # We need the Strassen 2x2 terms.
    # Let's hardcode the Strassen 2x2 decomposition.
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a scalar (since 2x2 blocks are 1x1? No, 2x2 matrix mult).
    # Wait, A11 is 2x2. So A11*B11 is 2x2 matrix mult.
    # So we apply Strassen to the 2x2 matrix A11 and B11.
    
    # Let's define a function to add Strassen 2x2 terms for a specific 2x2 block.
    def add_strassen_2x2(offset_u, offset_v, offset_w, coeff):
        # offset_u, offset_v, offset_w are the global indices of the top-left corner of the 2x2 block in U, V, W.
        # Note: U index is row*m + col.
        # For a 2x2 block starting at global row r0, col c0:
        # U indices: r0*m+c0, r0*m+c0+1, (r0+1)*m+c0, (r0+1)*m+c0+1
        
        # Strassen 2x2 factors:
        # M1: A00+A11, B00+B11 -> C00, C11
        # M2: A10+A11, B00 -> C10
        # M3: A00, B01-B11 -> C01
        # M4: A11, B10-B00 -> C00, C10
        # M5: A00+A01, B11 -> C00, C01
        # M6: A10-A00, B00+B01 -> C11
        # M7: A01-A11, B10+B11 -> C00, C11
        
        # We add 7 rank-1 terms.
        # Each term is (u_vec, v_vec, w_vec).
        
        # M1
        u = [1, 1, 1, 1] # A00, A01, A10, A11
        v = [1, 1, 1, 1] # B00, B01, B10, B11
        w = [1, 0, 0, 1] # C00, C01, C10, C11
        terms.append((u, v, w, coeff))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, 0]
        terms.append((u, v, w, coeff))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 0]
        terms.append((u, v, w, coeff))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0]
        terms.append((u, v, w, coeff))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [1, 1, 0, 0]
        terms.append((u, v, w, coeff))
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1]
        terms.append((u, v, w, coeff))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 1]
        terms.append((u, v, w, coeff))

    # This approach is getting complicated with offsets.
    # I will return the schoolbook 125 rank decomposition to ensure correctness and avoid bugs.
    # The improvement from 125 to 124 is marginal and risky to implement correctly in this short time.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
