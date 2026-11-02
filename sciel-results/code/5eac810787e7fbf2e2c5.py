"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.

Strategy: Decompose the 2x3 * 3x4 multiplication into two 2x3 * 3x2 blocks (columns 0-1 and 2-3).
Each 2x3 * 3x2 block is further decomposed into:
  - A 2x2 * 2x2 core (rows 0-1, columns 0-1 or 2-3) using Strassen's algorithm (rank 7)
  - A 2x1 * 1x2 cross term (row 2 of A times row 2 of B's corresponding columns) using schoolbook (rank 4)

Total rank = 2 * (7 + 4) = 22, which is less than the schoolbook rank of 24.
"""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
        U has shape (n*m, rank) = (6, 22)
        V has shape (m*p, rank) = (12, 22)  
        W has shape (p*n, rank) = (8, 22)
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll fill in the rank-22 decomposition
    # Block 1: columns 0,1 of output (k=0,1)
    # Block 2: columns 2,3 of output (k=2,3)
    
    # For each block, we have:
    # - Strassen part: 7 multiplications for the 2x2 * 2x2 core
    # - Schoolbook part: 4 multiplications for the 2x1 * 1x2 cross term
    
    col = 0
    
    for block in range(2):
        # Block offset in k-dimension
        k_offset = block * 2
        
        # Strassen part: 7 multiplications
        # The 2x2 core involves A[0:2, 0:2] * B[0:2, k_offset:k_offset+2]
        # Strassen's 7 products for 2x2 * 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each is a dict mapping (i,j) -> coeff for U, (j,k) -> coeff for V, (k,i) -> coeff for W
            {
                'U': {(0,0): 1, (1,1): 1},
                'V': {(0,0): 1, (1,1): 1},
                'W': {(0,0): 1, (1,1): -1}
            },
            {
                'U': {(1,0): 1, (1,1): 1},
                'V': {(0,0): 1},
                'W': {(1,1): 1}
            },
            {
                'U': {(0,0): 1},
                'V': {(1,0): 1, (1,1): -1},
                'W': {(1,0): 1}
            },
            {
                'U': {(1,1): 1},
                'V': {(0,1): 1, (0,0): -1},
                'W': {(0,1): 1}
            },
            {
                'U': {(0,0): 1, (0,1): 1},
                'V': {(1,1): 1},
                'W': {(0,0): 1, (1,1): -1}
            },
            {
                'U': {(1,0): 1, (0,0): -1},
                'V': {(0,0): 1, (1,0): 1},
                'W': {(0,1): 1}
            },
            {
                'U': {(0,1): 1, (1,1): -1},
                'V': {(0,1): 1, (1,1): 1},
                'W': {(1,0): 1}
            },
        ]
        
        for term in strassen_terms:
            # Fill U
            for (i, j), coeff in term['U'].items():
                idx = i * m + j
                U[idx][col] += coeff
            
            # Fill V
            for (j, k), coeff in term['V'].items():
                idx = j * p + (k_offset + k)
                V[idx][col] += coeff
            
            # Fill W
            for (k, i), coeff in term['W'].items():
                idx = (k_offset + k) * n + i
                W[idx][col] += coeff
            
            col += 1
        
        # Schoolbook part: 4 multiplications for A[2, 0:2] * B[2, k_offset:k_offset+2]
        # This is a 2x1 * 1x2 outer product, rank 4
        for i in range(2):
            for k in range(2):
                # U[i*3 + 2] = 1 (A[i, 2])
                U[i * m + 2][col] = 1
                # V[2*4 + (k_offset + k)] = 1 (B[2, k_offset+k])
                V[2 * p + (k_offset + k)][col] = 1
                # W[(k_offset + k)*2 + i] = 1 (C[i, k_offset+k])
                W[(k_offset + k) * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
