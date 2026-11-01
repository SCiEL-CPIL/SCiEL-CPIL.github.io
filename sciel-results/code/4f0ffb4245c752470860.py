import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    This uses a block decomposition:
    1. A1 (2x1) * B1 (1x4) : Rank 8
    2. A2 (2x2) * B2a (2x2) : Rank 7 (Strassen)
    3. A2 (2x2) * B2b (2x2) : Rank 7 (Strassen)
    Total Rank: 22
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def set_term(r, u_map, v_map, w_map):
        for idx, val in u_map.items():
            U[idx][r] += val
        for idx, val in v_map.items():
            V[idx][r] += val
        for idx, val in w_map.items():
            W[idx][r] += val

    # Indices for A (2x3): A[i][k] -> i*m + k
    # A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    
    # Indices for B (3x4): B[k][j] -> k*p + j
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # B20=8, B21=9, B22=10, B23=11
    
    # Indices for C (2x4): C[i][j] -> j*n + i (Trace convention W[k*n+i, r] in problem desc? 
    # Problem says W shape (p*n, rank). "official trace convention W[k*n+i,r]".
    # Usually C is n x p. If W is p x n, then W[j][i] corresponds to C[i][j]?
    # Let's check the error "target coordinate (0, 1, 2)".
    # If C is 2x4, indices are (i,j) with i in 0..1, j in 0..3.
    # If W is 4x2, index is j*2 + i.
    # C[0][0] -> 0. C[0][1] -> 2. C[0][2] -> 4. C[0][3] -> 6.
    # C[1][0] -> 1. C[1][1] -> 3. C[1][2] -> 5. C[1][3] -> 7.
    
    # Part 1: A1 (col 0) * B1 (row 0)
    # A1: A00 (0), A10 (3)
    # B1: B00 (0), B01 (1), B02 (2), B03 (3)
    
    # C00 = A00*B00
    set_term(0, {0: 1}, {0: 1}, {0: 1})
    # C01 = A00*B01
    set_term(1, {0: 1}, {1: 1}, {2: 1})
    # C02 = A00*B02
    set_term(2, {0: 1}, {2: 1}, {4: 1})
    # C03 = A00*B03
    set_term(3, {0: 1}, {3: 1}, {6: 1})
    # C10 = A10*B00
    set_term(4, {3: 1}, {0: 1}, {1: 1})
    # C11 = A10*B01
    set_term(5, {3: 1}, {1: 1}, {3: 1})
    # C12 = A10*B02
    set_term(6, {3: 1}, {2: 1}, {5: 1})
    # C13 = A10*B03
    set_term(7, {3: 1}, {3: 1}, {7: 1})

    # Part 2a: A2 (cols 1,2) * B2a (rows 1,2, cols 0,1)
    # A2: A01 (1), A02 (2), A11 (4), A12 (5)
    # B2a: B10 (4), B11 (5), B20 (8), B21 (9)
    # C2a: C00 (0), C01 (2), C10 (1), C11 (3)
    
    # Strassen M1: (A01+A12)(B10+B21) -> C00, C11
    set_term(8, {1: 1, 5: 1}, {4: 1, 9: 1}, {0: 1, 3: 1})
    
    # Strassen M2: (A11+A12)B10 -> C10, C11(-1)
    set_term(9, {4: 1, 5: 1}, {4: 1}, {1: 1, 3: -1})
    
    # Strassen M3: A01(B11-B21) -> C01, C11
    set_term(10, {1: 1}, {5: 1, 9: -1}, {2: 1, 3: 1})
    
    # Strassen M4: A12(B20-B10) -> C00(-1), C10
    set_term(11, {5: 1}, {8: 1, 4: -1}, {0: -1, 1: 1})
    
    # Strassen M5: (A01+A02)B21 -> C00(-1), C01
    set_term(12, {1: 1, 2: 1}, {9: 1}, {0: -1, 2: 1})
    
    # Strassen M6: (A11-A01)(B10+B11) -> C11(-1)
    set_term(13, {4: 1, 1: -1}, {4: 1, 5: 1}, {3: -1})
    
    # Strassen M7: (A02-A12)(B20+B21) -> C00
    set_term(14, {2: 1, 5: -1}, {8: 1, 9: 1}, {0: 1})

    # Part 2b: A2 (cols 1,2) * B2b (rows 1,2, cols 2,3)
    # A2: A01 (1), A02 (2), A11 (4), A12 (5)
    # B2b: B12 (6), B13 (7), B22 (10), B23 (11)
    # C2b: C02 (4), C03 (6), C12 (5), C13 (7)
    
    # Strassen M8: (A01+A12)(B12+B23) -> C02, C13
    set_term(15, {1: 1, 5: 1}, {6: 1, 11: 1}, {4: 1, 7: 1})
    
    # Strassen M9: (A11+A12)B12 -> C12, C13(-1)
    set_term(16, {4: 1, 5: 1}, {6: 1}, {5: 1, 7: -1})
    
    # Strassen M10: A01(B13-B23) -> C03, C13
    set_term(17, {1: 1}, {7: 1, 11: -1}, {6: 1, 7: 1})
    
    # Strassen M11: A12(B22-B12) -> C02(-1), C12
    set_term(18, {5: 1}, {10: 1, 6: -1}, {4: -1, 5: 1})
    
    # Strassen M12: (A01+A02)B23 -> C02(-1), C03
    set_term(19, {1: 1, 2: 1}, {11: 1}, {4: -1, 6: 1})
    
    # Strassen M13: (A11-A01)(B12+B13) -> C13(-1)
    set_term(20, {4: 1, 1: -1}, {6: 1, 7: 1}, {7: -1})
    
    # Strassen M14: (A02-A12)(B22+B23) -> C02
    set_term(21, {2: 1, 5: -1}, {10: 1, 11: 1}, {4: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
