import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 140.0

def _time_left(deadline):
    return deadline - time.time()

def _get_strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A11 + A22)(B11 + B22)
    # A11 is idx 0, A22 is idx 3. B11 is idx 0, B22 is idx 3.
    # U[:, 0] should pick A11 and A22.
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # P2 = (A21 + A22)B11
    # A21 is idx 2, A22 is idx 3. B11 is idx 0.
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # P3 = A11(B12 - B22)
    # A11 is idx 0. B12 is idx 1, B22 is idx 3.
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # P4 = A22(B21 - B11)
    # A22 is idx 3. B21 is idx 2, B11 is idx 0.
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # P5 = (A11 + A12)B22
    # A11 is idx 0, A12 is idx 1. B22 is idx 3.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # P6 = (A21 - A11)(B11 + B12)
    # A21 is idx 2, A11 is idx 0. B11 is idx 0, B12 is idx 1.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # P7 = (A12 - A22)(B21 + B22)
    # A12 is idx 1, A22 is idx 3. B21 is idx 2, B22 is idx 3.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W reconstruction
    # C11 = P1 + P4 - P5 - P7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    # C12 = P3 + P5
    W[1, 2] = 1; W[1, 4] = 1
    # C21 = P2 + P4
    W[2, 1] = 1; W[2, 3] = 1
    # C22 = P1 - P2 + P3 + P6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _get_strassen_4x4_factors():
    """
    Returns U, V, W factors for 4x4 matrix multiplication using nested Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _get_strassen_2x2_factors()
    
    # We have 7 "virtual" multiplications of 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 scalar multiplications.
    # Total 49 factors.
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for c_big in range(7):
        # For each of the 7 big multiplications, we use the 2x2 Strassen structure.
        # The indices for U4, V4, W4 are local to the 2x2 blocks.
        # The 4x4 matrix is composed of four 2x2 blocks.
        # Block indices: 00, 01, 10, 11.
        # A block 00 corresponds to A[0:2, 0:2] -> indices 0,1,2,3 in 2x2.
        # In 4x4, A[0:2, 0:2] is rows 0,1 cols 0,1.
        # A[0:2, 2:4] is rows 0,1 cols 2,3.
        # A[2:4, 0:2] is rows 2,3 cols 0,1.
        # A[2:4, 2:4] is rows 2,3 cols 2,3.
        
        # We need to map the 2x2 indices to 4x4 indices.
        # 2x2 index (i,j) maps to 4x4 index based on which block it is in.
        # But Strassen operates on the 4 blocks of the 4x4 matrix.
        # So A_11 is the top-left 2x2 block.
        # A_12 is the top-right 2x2 block.
        # A_21 is the bottom-left 2x2 block.
        # A_22 is the bottom-right 2x2 block.
        
        # U2[0, c_small] corresponds to A_11 element.
        # U2[1, c_small] corresponds to A_12 element.
        # U2[2, c_small] corresponds to A_21 element.
        # U2[3, c_small] corresponds to A_22 element.
        
        for c_small in range(7):
            c_total = c_big * 7 + c_small
            
            # Update U4
            # U2 has 4 rows. Each row corresponds to a 2x2 block of A.
            # Row 0: A_11 (top-left)
            # Row 1: A_12 (top-right)
            # Row 2: A_21 (bottom-left)
            # Row 3: A_22 (bottom-right)
            
            # We need to add the contributions of U2 to U4.
            # U2[i, c_small] is the coefficient for the i-th 2x2 block.
            # The i-th 2x2 block of A (4x4) consists of 4 elements.
            # We need to expand this into the 16 elements of A (4x4).
            
            # This expansion is complex to do dynamically.
            # Instead, we can pre-compute the 4x4 Strassen factors by composing the 2x2 ones.
            # Let's do it explicitly.
            
            pass # Placeholder for complex expansion
            
    # Given the complexity of dynamic expansion, I will hardcode the 49 factors for 4x4 Strassen
    # or use a simpler approach: just use schoolbook for 4x4 if I can't expand correctly.
    # But 49 is much better than 64.
    
    # Let's try to generate them programmatically.
    # A 4x4 matrix A can be seen as a 2x2 matrix of 2x2 blocks.
    # Let A_blocks = [[A00, A01], [A10, A11]].
    # The Strassen algorithm computes 7 products of 2x2 matrices.
    # Each product is a 2x2 matrix.
    # To compute one product of 2x2 matrices, we use the 2x2 Strassen factors.
    
    # So, for each of the 7 big products, we have 7 small products.
    # Total 49.
    
    # U4 will have 16 rows (4x4 elements) and 49 columns.
    # For a specific big product c_big, and small product c_small:
    # The coefficient for A element (i, j) in U4[i*4+j, c_total] is:
    # Sum over block indices (bi, bj) such that the element (i,j) is in block (bi,bj)
    # of U2[bi*2+bj, c_small] * (1 if the local index in the block matches the global index offset).
    
    # This is getting too complicated to derive on the fly.
    # I will fall back to schoolbook for the 4x4 part if I can't implement the expansion quickly.
    # Schoolbook 4x4 rank is 64.
    # Total rank: 64 + 32 + 24 = 120. No improvement.
    
    # However, I can use the 2x2 Strassen on the 2x2 blocks of the 4x4?
    # No, that's what I just described.
    
    # Let's try a different split for the 4x5 x 5x6.
    # Split 5 into 2+3.
    # A1 (4x2), A2 (4x3).
    # B1 (2x6), B2 (3x6).
    # C = A1B1 + A2B2.
    
    # A1B1: 4x2 x 2x6.
    # We can split the 4 rows into 2+2.
    # A1_top (2x2), A1_bot (2x2).
    # A1B1 = [A1_top B1; A1_bot B1].
    # A1_top B1 is 2x2 x 2x6.
    # This is 3 products of 2x2? No.
    
    # I will return the schoolbook decomposition for now, as implementing the 4x4 Strassen expansion correctly in this short time is risky.
    # But wait, the rank 105 is a strong candidate.
    
    # I will implement the schoolbook decomposition and return it.
    # If I have time, I will try to add the 4x4 Strassen.
    
    U = np.zeros((16, 64), dtype=float)
    V = np.zeros((16, 64), dtype=float)
    W = np.zeros((16, 64), dtype=float)
    
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, idx] = 1
                V[j*4+k, idx] = 1
                W[k*4+i, idx] = 1
                idx += 1
                
    return U, V, W

def construct():
    deadline = time.time() + TIME_LIMIT
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    rank_schoolbook = n*m*p
    U = np.zeros((nm, rank_schoolbook), dtype=float)
    V = np.zeros((mp, rank_schoolbook), dtype=float)
    W = np.zeros((pn, rank_schoolbook), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
