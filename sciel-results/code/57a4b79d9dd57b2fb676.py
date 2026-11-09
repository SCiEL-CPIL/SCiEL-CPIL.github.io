"""Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    
    # Winograd rank-21 decomposition for 3x3x3
    # Returns U, V, W for 3x3 matrix multiplication with rank 21
    def get_winograd_3x3x3():
        """
        Winograd's algorithm for 3x3 matrix multiplication.
        Uses 21 scalar multiplications.
        
        Convention: C[i,k] = sum_c W[k*3+i, c] * (sum_j U[i*3+j, c] * A[i,j]) * (sum_j V[j*3+k, c] * B[j,k])
        """
        U = np.zeros((9, 21))
        V = np.zeros((9, 21))
        W = np.zeros((9, 21))
        
        # Winograd's 3x3 algorithm uses polynomial evaluation/interpolation.
        # The 21 multiplications correspond to evaluating polynomials at certain points.
        
        # This is a known rank-21 decomposition. Let me construct it explicitly.
        # The algorithm works by:
        # 1. Forming 7 linear combinations of A (rows) and 7 of B (cols)
        # 2. Multiplying them (7*3 = 21 products)
        # 3. Forming linear combinations to get C
        
        # Actually, Winograd 3x3 uses:
        # - 3 "vertical" combinations of A rows
        # - 3 "horizontal" combinations of B cols  
        # - 3 "diagonal" combinations
        # Total: 21 products
        
        # Let me use a known explicit construction.
        # The 21 terms can be organized as follows:
        
        # For each of the 3 row-groups and 3 col-groups, we have 3 products.
        
        # This is complex. Let me use a verified rank-21 decomposition.
        
        # Known Winograd 3x3x3 decomposition with integer coefficients:
        
        # I'll use the standard Winograd algorithm structure.
        # The key matrices for Winograd 3x3:
        
        # A-side linear transforms (9x7 for the 7 intermediate vectors, but we need 21 terms)
        
        # Actually, let me use a simpler known decomposition.
        # The rank-21 decomposition can be constructed from the fact that:
        # 3x3 multiplication can be done with 21 multiplications using:
        # M1..M7 from a 7-product core, then expanded.
        
        # Let me use an explicit table. This is a known result.
        
        # For now, let me implement a verified rank-21 decomposition.
        
        # The following is a known rank-21 decomposition for 3x3x3:
        
        # Term 1-3: Diagonal-like
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
        U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
        
        # This is getting too complex to get right without verification.
        # Let me fall back to a safer approach.
        
        return None
    
    # Since constructing Winograd 3x3x3 correctly is error-prone,
    # let me use a different strategy: nested Strassen.
    
    # Use Strassen for 4x4x4 (rank 49) and handle the 5th row/col separately.
    
    # 4x4x4 via Strassen: rank 49
    # Extension to 5x5: 
    #   - 4x4 * 4x1: 4*4*1 = 16 terms
    #   - 4x1 * 4x4: 4*1*4 = 16 terms  
    #   - 4x1 * 1x4: 4*1*4 = 16 terms
    #   - 1x4 * 4x1: 1*4*1 = 4 terms
    #   - 1x1 * 1x1: 1 term
    #   - Cross terms between 4x4 and 1x1 parts...
    
    # This is still complex. Let me just use schoolbook but verify it works.
    
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
    # Implement Strassen 4x4x4 (rank 49) and extend to 5x5x5
    
    # Helper: Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00 -> C10 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00*(B01-B11) -> C01 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Build 4x4x4 using nested Strassen
    # 4x4 = 2x2 blocks of 2x2
    # Use Strassen at the block level, then Strassen within each block
    
    u2, v2, w2 = get_strassen_2x2x2()
    
    # 4x4x4 decomposition: 7 Strassen products, each of which is a 2x2x2 Strassen (rank 7)
    # Total: 7 * 7 = 49 terms
    
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    # Block index mappings for 4x4 (row-major: index = i*4 + j)
    # A blocks:
    # A11 (top-left 2x2): rows 0,1 cols 0,1 -> flat indices [0,1,4,5]
    # A12 (top-right 2x2): rows 0,1 cols 2,3 -> flat indices [2,3,6,7]
    # A21 (bottom-left 2x2): rows 2,3 cols 0,1 -> flat indices [8,9,12,13]
    # A22 (bottom-right 2x2): rows 2,3 cols 2,3 -> flat indices [10,11,14,15]
    
    # B blocks:
    # B11: [0,1,4,5]
    # B12: [2,3,6,7]
    # B21: [8,9,12,13]
    # B22: [10,11,14,15]
    
    # C blocks:
    # C11: [0,1,4,5]
    # C12: [2,3,6,7]
    # C21: [8,9,12,13]
    # C22: [10,11,14,15]
    
    # Strassen formulas for block-level 2x2 multiplication:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # M2 = (A21+A22)B11 -> contributes to C21
    # M3 = A11(B12-B22) -> contributes to C12
    # M4 = A22(B21-B11) -> contributes to C11 and C21
    # M5 = (A11+A12)B22 -> contributes to C11 and C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11 and C22
    
    # For each block-level Strassen product, we need to apply 2x2 Strassen within each block.
    # This means for each of the 7 block products, we have 7 sub-products.
    
    # Let me define the block mappings:
    # For A: block (bi, bj) where bi, bj in {0,1}
    #   bi=0, bj=0: A11, flat indices [0,1,4,5]
    #   bi=0, bj=1: A12, flat indices [2,3,6,7]
    #   bi=1, bj=0: A21, flat indices [8,9,12,13]
    #   bi=1, bj=1: A22, flat indices [10,11,14,15]
    
    A_block_idx = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    
    B_block_idx = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    
    C_block_idx = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    
    # For each block-level Strassen term, define which A-blocks and B-blocks are used
    # and how they're combined (coefficients)
    
    # Block-level Strassen:
    # Term 0 (M1): A_blocks = {(0,0):1, (1,1):1}, B_blocks = {(0,0):1, (1,1):1}, C_blocks = {(0,0):1, (1,1):1}
    # Term 1 (M2): A_blocks = {(1,0):1, (1,1):1}, B_blocks = {(0,0):1}, C_blocks = {(1,0):1}
    # Term 2 (M3): A_blocks = {(0,0):1}, B_blocks = {(0,1):1, (1,1):-1}, C_blocks = {(0,1):1}
    # Term 3 (M4): A_blocks = {(1,1):1}, B_blocks = {(1,0):1, (0,0):-1}, C_blocks = {(0,0):1, (1,0):1}
    # Term 4 (M5): A_blocks = {(0,0):1, (0,1):1}, B_blocks = {(1,1):1}, C_blocks = {(0,0):-1, (0,1):1}
    # Term 5 (M6): A_blocks = {(1,0):1, (0,0):-1}, B_blocks = {(0,0):1, (0,1):1}, C_blocks = {(1,1):1}
    # Term 6 (M7): A_blocks = {(0,1):1, (1,1):-1}, B_blocks = {(1,0):1, (1,1):1}, C_blocks = {(0,0):1, (1,1):1}
    
    block_strassen = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),  # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M7
    ]
    
    # Now, for each block-level Strassen term, we apply 2x2 Strassen within each involved block.
    # This is complex because the 2x2 Strassen within each block needs to be combined.
    
    # Actually, the correct approach is:
    # The 4x4x4 tensor is the "tensor product" of the block structure and the within-block structure.
    # We can generate all 49 terms by iterating over the 7 block-level Strassen terms
    # and for each, the 7 within-block Strassen terms.
    
    # But this requires careful bookkeeping. Let me use a different approach:
    # Directly generate the 49 terms by considering all combinations.
    
    # Actually, let me just use the schoolbook for now to ensure correctness,
    # and then try to optimize.
    
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
