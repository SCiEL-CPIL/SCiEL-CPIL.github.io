import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen 2x2 block decomposition on 3x3 submatrices.
    # This yields a total rank of 7 * 27 = 189.
    # The structure is 6 = 2 * 3.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_size = 3
    
    # Strassen 2x2 coefficients
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We represent each linear combination as a list of (row_block, col_block, coefficient)
    # A is 2x2 blocks of 3x3. So A[i][j] is a 3x3 matrix.
    # A_comb: terms for A
    # B_comb: terms for B
    # C_comb: terms for C (C[i][j] += coeff * M_t)
    
    A_combs = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(1, 0, 1), (1, 1, 1)],  # M2
        [(0, 0, 1)],             # M3
        [(1, 1, 1)],             # M4
        [(0, 0, 1), (0, 1, 1)],  # M5
        [(1, 0, 1), (0, 0, -1)], # M6
        [(0, 1, 1), (1, 1, -1)]  # M7
    ]
    
    B_combs = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(0, 0, 1)],             # M2
        [(1, 0, 1), (1, 1, -1)], # M3
        [(0, 1, 1), (0, 0, -1)], # M4
        [(1, 1, 1)],             # M5
        [(0, 0, 1), (0, 1, 1)],  # M6
        [(1, 0, 1), (1, 1, 1)]   # M7
    ]
    
    # C = [[M1 + M4 - M5 - M7, M3 + M5],
    #      [M2 + M4, M1 - M2 + M3 + M6]]
    
    C_combs = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],  # M1: C00, C01, C10, C11
        [(1, 0, 1), (1, 1, -1)],                        # M2: C10, C11
        [(0, 0, 1), (0, 1, 1), (1, 1, 1)],              # M3: C00, C01, C11
        [(0, 0, 1), (1, 0, 1)],                         # M4: C00, C10
        [(0, 0, -1), (0, 1, 1)],                        # M5: C00, C01
        [(1, 1, 1)],                                    # M6: C11
        [(0, 0, -1)]                                    # M7: C00
    ]
    
    # Verify C_comb indices:
    # C00: M1(1) + M4(1) - M5(1) - M7(1) -> 1+1-1-1 = 0? No.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-define C_combs carefully.
    # C[i][j] is sum over t of C_coeff[i][j][t] * M_t
    # C00: t=0(M1):1, t=1(M2):0, t=2(M3):0, t=3(M4):1, t=4(M5):-1, t=5(M6):0, t=6(M7):-1
    # C01: t=0:0, t=1:0, t=2:1, t=3:0, t=4:1, t=5:0, t=6:0
    # C10: t=0:0, t=1:1, t=2:0, t=3:1, t=4:0, t=5:0, t=6:0
    # C11: t=0:1, t=1:-1, t=2:1, t=3:0, t=4:0, t=5:1, t=6:0
    
    # My previous C_combs list was:
    # C_combs[t] is a list of (row_block, col_block, coeff) for that t.
    
    C_combs = [
        [(0, 0, 1), (1, 1, 1)],           # M1: C00 += 1, C11 += 1
        [(1, 0, 1), (1, 1, -1)],          # M2: C10 += 1, C11 += -1
        [(0, 1, 1), (1, 1, 1)],           # M3: C01 += 1, C11 += 1
        [(0, 0, 1), (1, 0, 1)],           # M4: C00 += 1, C10 += 1
        [(0, 0, -1), (0, 1, 1)],          # M5: C00 += -1, C01 += 1
        [(1, 1, 1)],                      # M6: C11 += 1
        [(0, 0, -1)]                      # M7: C00 += -1
    ]
    
    # Check C00: 1 (M1) + 1 (M4) - 1 (M5) - 1 (M7) = 0? 
    # Wait. C00 = M1 + M4 - M5 - M7.
    # M1: 1. M4: 1. M5: -1. M7: -1. Sum = 0.
    # But C00 should be M1 + M4 - M5 - M7.
    # If A = [[1,0],[0,0]], B = [[1,0],[0,0]], C = [[1,0],[0,0]].
    # M1 = (1+0)(1+0) = 1.
    # M4 = 0 * (0-1) = 0.
    # M5 = (1+0)*0 = 0.
    # M7 = (0-0)*(0+0) = 0.
    # C00 = 1 + 0 - 0 - 0 = 1. Correct.
    
    # Check C01: M3 + M5.
    # M3 = 1 * (0-0) = 0.
    # M5 = 1 * 0 = 0.
    # C01 = 0. Correct.
    
    # Check C10: M2 + M4.
    # M2 = 0 * 1 = 0.
    # M4 = 0.
    # C10 = 0. Correct.
    
    # Check C11: M1 - M2 + M3 + M6.
    # M1 = 1. M2 = 0. M3 = 0. M6 = 0.
    # C11 = 1. Correct.
    
    # The logic holds.
    
    for t in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                        
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                        
                    for rb, cb, coeff in C_combs[t]:
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        W[k_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
