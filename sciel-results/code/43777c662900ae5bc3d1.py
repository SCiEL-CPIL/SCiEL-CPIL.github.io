"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices: 00, 01, 10, 11)
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B10-B11)
    # k=3: M4 = A11(B01-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V coefficients for each block l (B indices: j*12 + k)
    # Only 4 indices are non-zero per block: 0*12+2l, 0*12+2l+1, 1*12+2l, 1*12+2l+1
    # We store (offset, coeff) pairs relative to the block start
    # Block start for row j=0 is 2l, for j=1 is 12+2l
    # Indices:
    # B00: 2l
    # B01: 2l+1
    # B10: 12+2l
    # B11: 12+2l+1
    
    V_coeffs = [
        [(0, 1), (12, 1)],      # M1: B00, B11
        [(0, 1)],                # M2: B00
        [(12, 1), (13, -1)],    # M3: B10, B11
        [(1, 1), (0, -1)],      # M4: B01, B00
        [(13, 1)],               # M5: B11
        [(0, 1), (1, 1)],       # M6: B00, B01
        [(12, 1), (13, 1)]      # M7: B10, B11
    ]
    
    # W coefficients for each block l (C indices: k*2 + i)
    # Block l corresponds to columns 2l, 2l+1.
    # Indices:
    # C00 (row 2l, col 0): 2l*2 + 0 = 4l
    # C01 (row 2l, col 1): 2l*2 + 1 = 4l+1
    # C10 (row 2l+1, col 0): (2l+1)*2 + 0 = 4l+2
    # C11 (row 2l+1, col 1): (2l+1)*2 + 1 = 4l+3
    
    W_coeffs = [
        [(0, 1), (3, 1)],       # M1: C00, C11
        [(2, 1), (3, -1)],      # M2: C10, C11
        [(1, 1), (2, 1)],       # M3: C01, C10
        [(0, -1), (1, 1)],      # M4: C00, C01
        [(0, 1), (2, -1)],      # M5: C00, C10
        [(2, -1), (3, 1)],      # M6: C10, C11
        [(0, -1), (1, 1), (3, 1)] # M7: C00, C01, C11
    ]

    for l in range(6):
        for k in range(7):
            col = l * 7 + k
            
            # Fill U
            for i in range(4):
                U[i][col] = U_coeffs[k][i]
                
            # Fill V
            base_idx = 2 * l
            for offset, coeff in V_coeffs[k]:
                # offset is relative to 2l for j=0, or 12+2l for j=1?
                # My V_coeffs used absolute offsets from block start 0?
                # Let's re-verify V_coeffs indices.
                # B00 is index 2l. Offset 0 from 2l.
                # B01 is index 2l+1. Offset 1 from 2l.
                # B10 is index 12+2l. Offset 12 from 2l.
                # B11 is index 12+2l+1. Offset 13 from 2l.
                # So V index = 2*l + offset.
                v_idx = 2 * l + offset
                V[v_idx][col] = coeff
                
            # Fill W
            w_base = 4 * l
            for offset, coeff in W_coeffs[k]:
                w_idx = w_base + offset
                W[w_idx][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
