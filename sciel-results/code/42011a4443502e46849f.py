"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 2x2 by 2x11 matrix multiplication.
    
    Strategy:
    - Use Strassen's algorithm to decompose the 2x2 by 2x2 blocks, reducing the rank per block pair.
    - The naive rank is 2*2*11 = 44.
    - By pairing adjacent columns and using Strassen, we can reduce the rank.
    
    For 2x2 by 2x2, Strassen gives rank 7 instead of 8.
    For 2x2 by 2x11, we can pair columns (0,1), (2,3), (4,5), (6,7), (8,9), and handle column 10 separately.
    
    For each pair of columns, we use 7 terms. For the last single column, we use 4 terms (naive).
    Total rank: 5*7 + 4 = 35 + 4 = 39.
    
    Let me work out the details:
    - A is 2x2: A = [[a0, a1], [a2, a3]]
    - B is 2x11: B = [[b0, b1, ..., b10], [c0, c1, ..., c10]]
    - C = A*B is 2x11
    
    For columns (2k, 2k+1), we use Strassen:
    M1 = (a0+a3)*(b_{0,2k}+c_{1,2k+1})  # Wait, need to be careful with indexing
    
    Let me reconsider. For Strassen on 2x2 by 2x2:
    If B_block = [[b00, b01], [b10, b11]], then:
    M1 = (a0+a3)*(b00+b11)
    M2 = (a2+a3)*(b00+b01)
    M3 = (a0-a2)*(b10+b11)
    M4 = a3*(b11-b10)
    M5 = a0*(b00-b10)
    M6 = (a1+a3)*(b11-b01)
    M7 = (a1-a2)*(b00-b10)
    
    Then:
    c00 = M1 - M4 + M6 - M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M3 + M5 + M7
    
    For our case, pairing columns 2k and 2k+1:
    B_block for columns (2k, 2k+1) = [[b_{0,2k}, b_{0,2k+1}], [b_{1,2k}, b_{1,2k+1}]]
    
    So:
    M1 = (a0+a3)*(b_{0,2k}+b_{1,2k+1})
    M2 = (a2+a3)*(b_{0,2k}+b_{0,2k+1})
    M3 = (a0-a2)*(b_{1,2k}+b_{1,2k+1})
    M4 = a3*(b_{1,2k+1}-b_{1,2k})
    M5 = a0*(b_{0,2k}-b_{1,2k})
    M6 = (a1+a3)*(b_{1,2k+1}-b_{0,2k+1})
    M7 = (a1-a2)*(b_{0,2k}-b_{1,2k})
    
    Then:
    c_{0,2k} = M1 - M4 + M6 - M7
    c_{0,2k+1} = M3 + M5
    c_{1,2k} = M2 + M4
    c_{1,2k+1} = M1 - M3 + M5 + M7
    
    For the last column (column 10), we use naive:
    M8 = a0*b_{0,10}
    M9 = a1*b_{1,10}
    M10 = a2*b_{0,10}
    M11 = a3*b_{1,10}
    
    Then:
    c_{0,10} = M8 + M9
    c_{1,10} = M10 + M11
    
    Total rank: 5*7 + 4 = 35 + 4 = 39.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    rank = 39
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    # A indices: 0=a0, 1=a1, 2=a2, 3=a3
    # B indices: row 0: 0..10, row 1: 11..21
    # C indices: W[k*n+i, r] where k is column index, i is row index
    # So c_{0,k} = W[k*2+0, r] = W[2k, r]
    # c_{1,k} = W[k*2+1, r] = W[2k+1, r]
    
    col = 0
    # Process 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
    for k in range(5):
        # Column indices for this pair
        c0 = 2 * k
        c1 = 2 * k + 1
        
        # Strassen terms
        # M1 = (a0+a3)*(b_{0,c0}+b_{1,c1})
        U[0][col] = 1  # a0
        U[3][col] = 1  # a3
        V[0 * 11 + c0][col] = 1  # b_{0,c0}
        V[1 * 11 + c1][col] = 1  # b_{1,c1}
        W[2 * c0][col] = 1  # c_{0,c0} += M1
        W[2 * c1][col] = 1  # c_{0,c1} += M1
        col += 1
        
        # M2 = (a2+a3)*(b_{0,c0}+b_{0,c1})
        U[2][col] = 1  # a2
        U[3][col] = 1  # a3
        V[0 * 11 + c0][col] = 1  # b_{0,c0}
        V[0 * 11 + c1][col] = 1  # b_{0,c1}
        W[2 * c0 + 1][col] = 1  # c_{1,c0} += M2
        W[2 * c1 + 1][col] = 1  # c_{1,c1} += M2
        col += 1
        
        # M3 = (a0-a2)*(b_{1,c0}+b_{1,c1})
        U[0][col] = 1  # a0
        U[2][col] = -1  # a2
        V[1 * 11 + c0][col] = 1  # b_{1,c0}
        V[1 * 11 + c1][col] = 1  # b_{1,c1}
        W[2 * c0][col] = 1  # c_{0,c0} += M3
        W[2 * c0 + 1][col] = 1  # c_{1,c0} += M3
        col += 1
        
        # M4 = a3*(b_{1,c1}-b_{1,c0})
        U[3][col] = 1  # a3
        V[1 * 11 + c1][col] = 1  # b_{1,c1}
        V[1 * 11 + c0][col] = -1  # b_{1,c0}
        W[2 * c0][col] = -1  # c_{0,c0} -= M4
        W[2 * c0 + 1][col] = 1  # c_{1,c0} += M4
        col += 1
        
        # M5 = a0*(b_{0,c0}-b_{1,c0})
        U[0][col] = 1  # a0
        V[0 * 11 + c0][col] = 1  # b_{0,c0}
        V[1 * 11 + c0][col] = -1  # b_{1,c0}
        W[2 * c0][col] = 1  # c_{0,c0} += M5
        W[2 * c1][col] = 1  # c_{0,c1} += M5
        col += 1
        
        # M6 = (a1+a3)*(b_{1,c1}-b_{0,c1})
        U[1][col] = 1  # a1
        U[3][col] = 1  # a3
        V[1 * 11 + c1][col] = 1  # b_{1,c1}
        V[0 * 11 + c1][col] = -1  # b_{0,c1}
        W[2 * c0][col] = 1  # c_{0,c0} += M6
        W[2 * c1][col] = -1  # c_{0,c1} -= M6
        col += 1
        
        # M7 = (a1-a2)*(b_{0,c0}-b_{1,c0})
        U[1][col] = 1  # a1
        U[2][col] = -1  # a2
        V[0 * 11 + c0][col] = 1  # b_{0,c0}
        V[1 * 11 + c0][col] = -1  # b_{1,c0}
        W[2 * c0][col] = -1  # c_{0,c0} -= M7
        W[2 * c0 + 1][col] = 1  # c_{1,c0} += M7
        W[2 * c1][col] = 1  # c_{1,c1} += M7
        col += 1
    
    # Process last column (column 10) with naive multiplication
    c10 = 10
    # M8 = a0*b_{0,10}
    U[0][col] = 1  # a0
    V[0 * 11 + c10][col] = 1  # b_{0,10}
    W[2 * c10][col] = 1  # c_{0,10} += M8
    col += 1
    
    # M9 = a1*b_{1,10}
    U[1][col] = 1  # a1
    V[1 * 11 + c10][col] = 1  # b_{1,10}
    W[2 * c10][col] = 1  # c_{0,10} += M9
    col += 1
    
    # M10 = a2*b_{0,10}
    U[2][col] = 1  # a2
    V[0 * 11 + c10][col] = 1  # b_{0,10}
    W[2 * c10 + 1][col] = 1  # c_{1,10} += M10
    col += 1
    
    # M11 = a3*b_{1,10}
    U[3][col] = 1  # a3
    V[1 * 11 + c10][col] = 1  # b_{1,10}
    W[2 * c10 + 1][col] = 1  # c_{1,10} += M11
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
