"""Matrix multiplication 6,6,6 - Strassen 2x2 block decomposition."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time


def construct():
    """
    Construct a low-rank exact tensor decomposition for 6x6 matrix multiplication
    using Strassen's 2x2 block decomposition.
    
    We partition the 6x6 matrices into 3x3 blocks of size 2x2.
    Each 2x2 block multiplication requires 7 scalar multiplications (Strassen),
    so total rank = 3*3*7 = 63.
    
    The trace convention: W[k*n+i, r] where n=p=6.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen 7 formulas for 2x2 blocks:
    # Given A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = A*B = [[c11,c12],[c21,c22]]
    #
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b21-b22)
    # M4 = a22*(b12-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b21)
    # M7 = (a12-a22)*(b12+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Each formula: (U_coefficients, V_coefficients, W_coefficients)
    # U indices: (local_row, local_col) in the 2x2 block of A
    # V indices: (local_row, local_col) in the 2x2 block of B  
    # W indices: (local_col, local_row) in the 2x2 block of C (trace convention)
    
    # Formula: (A_part, B_part, C_part) where each part is a dict of {(r,c): coeff}
    strassen_formulas = [
        # M1 = (a11+a22)*(b11+b22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1, (0,1): 0, (1,0): 0}),
        # M2 = (a21+a22)*b11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1, (0,0): 0, (0,1): 0}),
        # M3 = a11*(b21-b22)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1, (0,0): 0, (1,0): 0}),
        # M4 = a22*(b12-b11)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1, (0,1): 0, (1,1): 0}),
        # M5 = (a11+a12)*b22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6 = (a21-a11)*(b11+b21)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,0): 0, (1,1): 1, (0,0): 0, (0,1): 0}),
        # M7 = (a12-a22)*(b12+b22)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (1,1): 1, (0,1): 0, (1,0): 0}),
    ]
    
    rank = num_blocks * num_blocks * 7  # 3*3*7 = 63
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for bi in range(num_blocks):  # block row of A / C
        for bj in range(num_blocks):  # block col of A / block row of B
            for bk in range(num_blocks):  # block col of B / C
                for (a_coeffs, b_coeffs, c_coeffs) in strassen_formulas:
                    for (ar, ac), a_val in a_coeffs.items():
                        if a_val != 0:
                            # Global row in A: bi*block_size + ar
                            # Global col in A: bj*block_size + ac
                            global_row = bi * block_size + ar
                            global_col = bj * block_size + ac
                            U[global_row * m + global_col, col] = a_val
                    
                    for (br, bc), b_val in b_coeffs.items():
                        if b_val != 0:
                            # Global row in B: bj*block_size + br
                            # Global col in B: bk*block_size + bc
                            global_row = bj * block_size + br
                            global_col = bk * block_size + bc
                            V[global_row * p + global_col, col] = b_val
                    
                    for (cr, cc), c_val in c_coeffs.items():
                        if c_val != 0:
                            # Global row in C: bi*block_size + cr
                            # Global col in C: bk*block_size + cc
                            global_row = bi * block_size + cr
                            global_col = bk * block_size + cc
                            # Trace convention: W[k*n + i, r] where k=col, i=row
                            W[global_col * n + global_row, col] = c_val
                    
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
