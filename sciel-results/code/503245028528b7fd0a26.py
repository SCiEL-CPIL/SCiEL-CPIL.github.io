import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Theoretical minimal rank for 2x4 x 4x6 is 42.
    # Construction:
    # A (2x4) split into A0 (2x2), A1 (2x2)
    # B (4x6) split into B0 (2x6), B1 (2x6)
    # B0 split into 3 blocks of 2x2: B0_0, B0_1, B0_2
    # B1 split into 3 blocks of 2x2: B1_0, B1_1, B1_2
    # C = A0 @ B0 + A1 @ B1
    # A0 @ B0 = A0 @ B0_0 + A0 @ B0_1 + A0 @ B0_2
    # Each 2x2 @ 2x2 has rank 7 (Strassen)
    # Total rank = 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # X (2x2) * Y (2x2) = Z (2x2)
    # Indices:
    # X: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Y: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Z: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U_coeffs[s] maps (i, j) to coeff for the s-th product
    # V_coeffs[s] maps (j, k) to coeff for the s-th product
    # W_coeffs[s] maps (i, k) to coeff for the s-th product
    
    # Standard Strassen:
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y10-y00)
    # P4 = x11(y01-y11)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01+x10)(y10+y11)
    
    # Z00 = P1 + P3 - P5 - P6
    # Z01 = P4 + P5
    # Z10 = P2 + P3
    # Z11 = P1 - P2 + P4 + P7
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # P1
        {(1, 0): 1, (1, 1): 1},      # P2
        {(0, 0): 1},                 # P3
        {(1, 1): 1},                 # P4
        {(0, 0): 1, (0, 1): 1},      # P5
        {(1, 0): 1, (0, 0): -1},     # P6
        {(0, 1): 1, (1, 0): 1}       # P7 -- Wait, P7 is (x01+x10)(y10+y11)
    ]
    # Correction on P7 U_coeffs: x01 is (0,1), x10 is (1,0).
    # So U_coeffs[6] should be {(0, 1): 1, (1, 0): 1}.
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # P1
        {(0, 0): 1},                 # P2
        {(1, 0): 1, (0, 0): -1},     # P3
        {(0, 1): 1, (1, 1): -1},     # P4
        {(1, 1): 1},                 # P5
        {(0, 0): 1, (0, 1): 1},      # P6
        {(1, 0): 1, (1, 1): 1}       # P7
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # P1: Z00+1, Z11+1
        {(1, 0): 1, (1, 1): -1},     # P2: Z10+1, Z11-1
        {(0, 0): 1, (1, 0): 1},      # P3: Z00+1, Z10+1
        {(0, 1): 1, (1, 1): 1},      # P4: Z01+1, Z11+1
        {(0, 0): -1, (0, 1): 1},     # P5: Z00-1, Z01+1
        {(0, 0): -1, (1, 1): 1},     # P6: Z00-1, Z11+1 -- Wait, Z00 = P1+P3-P5-P6. So P6 coeff for Z00 is -1. Z11 = P1-P2+P4+P7. P6 is not in Z11?
        {(0, 0): 1}                  # P7: Z11+1
    ]
    # Let's re-verify W_coeffs carefully.
    # Z00 = P1 + P3 - P5 - P6
    # Z01 = P4 + P5
    # Z10 = P2 + P3
    # Z11 = P1 - P2 + P4 + P7
    
    # P1: Z00(+1), Z11(+1) -> {(0,0):1, (1,1):1}
    # P2: Z10(+1), Z11(-1) -> {(1,0):1, (1,1):-1}
    # P3: Z00(+1), Z10(+1) -> {(0,0):1, (1,0):1}
    # P4: Z01(+1), Z11(+1) -> {(0,1):1, (1,1):1}
    # P5: Z00(-1), Z01(+1) -> {(0,0):-1, (0,1):1}
    # P6: Z00(-1)           -> {(0,0):-1}
    # P7: Z11(+1)           -> {(1,1):1}
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): -1, (0, 1): 1},
        {(0, 0): -1},
        {(1, 1): 1}
    ]

    col = 0
    for a_block in range(2): # 0 for A0 (cols 0-1), 1 for A1 (cols 2-3)
        for b_block in range(3): # 0 for B cols 0-1, 1 for B cols 2-3, 2 for B cols 4-5
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
