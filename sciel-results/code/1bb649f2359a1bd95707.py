"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x8 matrix.
    
    Uses Strassen's algorithm applied to 4 independent 2x2x2 blocks.
    Rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_A = [
        [1, 1, 0, 0, 1, 0, 0], # A00 coeff in M1..M7? No, M1 uses A00+A11
        # Let's define A_coefs[m_idx, r_idx]
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # M1: A00+A11 -> [1, 0, 0, 1]
        # M2: A10+A11 -> [0, 0, 1, 1]
        # M3: A00 -> [1, 0, 0, 0]
        # M4: A11 -> [0, 0, 0, 1]
        # M5: A00+A01 -> [1, 1, 0, 0]
        # M6: A10-A00 -> [-1, 0, 1, 0]
        # M7: A01-A11 -> [0, 1, 0, -1]
    ]
    A_coefs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    B_coefs = [
        [1, 0, 1, 0], # M1: B00+B11 (B indices: 00, 01, 10, 11 -> 0, 1, 2, 3)
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    C_coefs = [
        [1, 0, 0, 0], # M1: C00
        [0, 0, 0, 1], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 0, 1], # M4: C00, C10
        [-1, 1, 0, 0], # M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    # Wait, C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # M1: C00(1), C11(1)
    # M2: C10(1), C11(-1)
    # M3: C01(1), C11(1)
    # M4: C00(1), C10(1)
    # M5: C00(-1), C01(1)
    # M6: C11(1)
    # M7: C00(1)
    
    C_coefs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Fill U, V, W
    for block in range(4):
        for s in range(7):
            r = block * 7 + s
            
            # U: A part
            for ai in range(4):
                U[ai][r] = A_coefs[s][ai]
            
            # V: B part
            # B block 'block' uses columns 2*block, 2*block+1
            # B indices in V: j*p + k
            # j=0, k=2*block -> 0*8 + 2*block = 2*block
            # j=0, k=2*block+1 -> 2*block+1
            # j=1, k=2*block -> 1*8 + 2*block = 8 + 2*block
            # j=1, k=2*block+1 -> 8 + 2*block + 1
            b_indices = [2*block, 2*block+1, 8+2*block, 8+2*block+1]
            for bi_idx in range(4):
                V[b_indices[bi_idx]][r] = B_coefs[s][bi_idx]
            
            # W: C part
            # C block 'block' uses columns 2*block, 2*block+1
            # C indices in W: k*n + i
            # i=0, k=2*block -> 2*block*2 + 0 = 4*block
            # i=1, k=2*block -> 4*block + 1
            # i=0, k=2*block+1 -> (2*block+1)*2 + 0 = 4*block + 2
            # i=1, k=2*block+1 -> 4*block + 3
            c_indices = [4*block, 4*block+1, 4*block+2, 4*block+3]
            for ci_idx in range(4):
                W[c_indices[ci_idx]][r] = C_coefs[s][ci_idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
