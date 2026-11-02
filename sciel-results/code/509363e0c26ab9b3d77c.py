import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 decomposition
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # U_2x2[r][u, v] for A
    # V_2x2[r][w, z] for B
    # W_2x2[r][p, q] for C
    
    # Strassen 2x2x2
    U_2 = [
        [[1, 0], [1, 1]],  # M1: A00 + A11
        [[0, 0], [1, 1]],  # M2: A10 + A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00 + A01
        [[1, 0], [1, 0]],  # M6: A10 - A00 (Wait, A10 - A00 is [[-1, 0], [1, 0]]? No.)
        # M6: (A10 - A00) * (B00 + B01)
        # A10 is block (1,0), A00 is block (0,0).
        # So M6 uses A[1,0] - A[0,0].
        # In 2x2 matrix, A[1,0] is element (1,0). A[0,0] is element (0,0).
        # So U_2[5] should be [[-1, 0], [1, 0]]?
        # Let's re-verify Strassen indices.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # U_2 factors for A:
        # M1: A00+A11 -> [[1,0],[1,1]]? No, A00 is (0,0), A11 is (1,1).
        # So [[1,0],[0,1]]? No, A00 is element 0,0. A11 is element 1,1.
        # So U_2[0] = [[1,0],[0,1]]? No, that's identity.
        # A00 is at (0,0). A11 is at (1,1).
        # So U_2[0] = [[1,0],[0,1]]? No, that sums diagonal.
        # Wait, A is a 2x2 matrix.
        # A00 is A[0,0]. A11 is A[1,1].
        # So M1 = (A[0,0] + A[1,1]) * ...
        # So U_2[0] = [[1,0],[0,1]]? No, that's not right.
        # The factor is a 2x2 matrix of coefficients.
        # U_2[0][0,0] = 1, U_2[0][1,1] = 1. Others 0.
        
        # Let's list them properly.
    ]
    
    # Correct Strassen 2x2 factors
    # r=0: M1
    U_2[0] = [[1, 0], [0, 1]] # A00 + A11? No. A00 is (0,0). A11 is (1,1).
    # Wait, standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # A00 is top-left. A11 is bottom-right.
    # So U_2[0] = [[1, 0], [0, 1]]? No, that's identity.
    # Identity times B? No.
    # The product is (A00+A11) * (B00+B11).
    # This is not a matrix multiplication of A and B.
    # It is a scalar multiplication of sums?
    # No, in the block Strassen, A00, A11 are blocks.
    # So (A00 + A11) is a matrix. (B00 + B11) is a matrix.
    # So we are multiplying two matrices.
    # So the "U" factor for the block level is not a 2x2 matrix of coefficients for the entries of A.
    # It is a linear combination of the blocks.
    
    # My previous logic:
    # Level 1: 7 products of 2x2 matrices.
    # Level 2: Each 2x2 product is a 2x2x2 tensor.
    
    # So for each of the 7 Level 1 products, we need the 2x2x2 decomposition.
    # The 2x2x2 decomposition is Strassen (rank 7).
    
    # So for each r in 0..6:
    #   We have A_block = sum_u,v U1[r][u,v] A[u,v]
    #   We have B_block = sum_w,z V1[r][w,z] B[w,z]
    #   Then M_r = A_block * B_block.
    #   We decompose M_r = A_block * B_block using Strassen 2x2x2.
    #   This gives 7 scalar multiplications.
    
    # So total 49 scalar multiplications.
    
    pass
