"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Structure: 2 blocks of A (2x2 each) x 2 blocks of B (2x3 each)
    # Each (2x2) x (2x3) uses rank 11 (7 Strassen + 4 direct)
    # Total rank = 2 * 2 * 11 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b_A in range(2):
        for b_B in range(2):
            # Local indices
            # A: i in {0,1}, j_loc in {0,1} -> global j = 2*b_A + j_loc
            # B: j_loc in {0,1}, k_loc in {0,1,2} -> global j = 2*b_A + j_loc, global k = 3*b_B + k_loc
            # C: i in {0,1}, k_loc in {0,1,2} -> global k = 3*b_B + k_loc
            
            # --- Strassen Part (cols 0,1 of local B) ---
            # P1 = A[0,0] * (B[1,1] - B[1,0])
            # P2 = (A[0,0] + A[1,0]) * B[1,0]
            # P3 = (A[0,1] + A[1,1]) * B[0,0]
            # P4 = A[1,1] * (B[0,1] - B[0,0])
            # P5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            # P6 = (A[0,1] - A[0,0]) * (B[0,0] + B[1,0])
            # P7 = (A[1,0] - A[1,1]) * (B[0,1] + B[1,1])
            
            # Coefficients for U (A), V (B), W (C)
            # Order: P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11
            
            # P1
            U[0 * 4 + (2*b_A + 0)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 1)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 0)][col] = -1
            W[(3*b_B + 0) * 2 + 0][col] = -1
            W[(3*b_B + 0) * 2 + 1][col] = 1
            W[(3*b_B + 1) * 2 + 0][col] = 1
            W[(3*b_B + 1) * 2 + 1][col] = 0 # C01, C10, C11 contributions?
            # Strassen C:
            # C00 = P5 - P1 + P3 - P7
            # C01 = P4 + P5 - P2 + P6
            # C10 = P1 + P2 - P5 + P7
            # C11 = P3 + P4 - P6 + P5
            
            # Let's fill W for P1:
            # C00: -1, C01: 0, C10: 1, C11: 0
            W[(3*b_B + 0) * 2 + 0][col] = -1
            W[(3*b_B + 0) * 2 + 1][col] = 0
            W[(3*b_B + 1) * 2 + 0][col] = 1
            W[(3*b_B + 1) * 2 + 1][col] = 0
            col += 1
            
            # P2
            U[0 * 4 + (2*b_A + 0)][col] = 1
            U[1 * 4 + (2*b_A + 0)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 0)][col] = 1
            # C00: 0, C01: -1, C10: 1, C11: 0
            W[(3*b_B + 0) * 2 + 0][col] = 0
            W[(3*b_B + 0) * 2 + 1][col] = -1
            W[(3*b_B + 1) * 2 + 0][col] = 1
            W[(3*b_B + 1) * 2 + 1][col] = 0
            col += 1
            
            # P3
            U[0 * 4 + (2*b_A + 1)][col] = 1
            U[1 * 4 + (2*b_A + 1)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 0)][col] = 1
            # C00: 1, C01: 0, C10: 0, C11: 1
            W[(3*b_B + 0) * 2 + 0][col] = 1
            W[(3*b_B + 0) * 2 + 1][col] = 0
            W[(3*b_B + 1) * 2 + 0][col] = 0
            W[(3*b_B + 1) * 2 + 1][col] = 1
            col += 1
            
            # P4
            U[1 * 4 + (2*b_A + 1)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 1)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 0)][col] = -1
            # C00: 0, C01: 1, C10: 0, C11: 1
            W[(3*b_B + 0) * 2 + 0][col] = 0
            W[(3*b_B + 0) * 2 + 1][col] = 1
            W[(3*b_B + 1) * 2 + 0][col] = 0
            W[(3*b_B + 1) * 2 + 1][col] = 1
            col += 1
            
            # P5
            U[0 * 4 + (2*b_A + 0)][col] = 1
            U[1 * 4 + (2*b_A + 1)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 0)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 1)][col] = 1
            # C00: 1, C01: 1, C10: -1, C11: 1
            W[(3*b_B + 0) * 2 + 0][col] = 1
            W[(3*b_B + 0) * 2 + 1][col] = 1
            W[(3*b_B + 1) * 2 + 0][col] = -1
            W[(3*b_B + 1) * 2 + 1][col] = 1
            col += 1
            
            # P6
            U[0 * 4 + (2*b_A + 1)][col] = 1
            U[0 * 4 + (2*b_A + 0)][col] = -1
            V[(2*b_A + 0) * 6 + (3*b_B + 0)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 0)][col] = 1
            # C00: 0, C01: 1, C10: 0, C11: -1
            W[(3*b_B + 0) * 2 + 0][col] = 0
            W[(3*b_B + 0) * 2 + 1][col] = 1
            W[(3*b_B + 1) * 2 + 0][col] = 0
            W[(3*b_B + 1) * 2 + 1][col] = -1
            col += 1
            
            # P7
            U[1 * 4 + (2*b_A + 0)][col] = 1
            U[1 * 4 + (2*b_A + 1)][col] = -1
            V[(2*b_A + 0) * 6 + (3*b_B + 1)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 1)][col] = 1
            # C00: -1, C01: 0, C10: 1, C11: 0
            W[(3*b_B + 0) * 2 + 0][col] = -1
            W[(3*b_B + 0) * 2 + 1][col] = 0
            W[(3*b_B + 1) * 2 + 0][col] = 1
            W[(3*b_B + 1) * 2 + 1][col] = 0
            col += 1
            
            # --- Direct Part (col 2 of local B) ---
            # P8 = A[0,0] * B[0,2]
            U[0 * 4 + (2*b_A + 0)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 2)][col] = 1
            W[(3*b_B + 2) * 2 + 0][col] = 1
            col += 1
            
            # P9 = A[0,1] * B[1,2]
            U[0 * 4 + (2*b_A + 1)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 2)][col] = 1
            W[(3*b_B + 2) * 2 + 0][col] = 1
            col += 1
            
            # P10 = A[1,0] * B[0,2]
            U[1 * 4 + (2*b_A + 0)][col] = 1
            V[(2*b_A + 0) * 6 + (3*b_B + 2)][col] = 1
            W[(3*b_B + 2) * 2 + 1][col] = 1
            col += 1
            
            # P11 = A[1,1] * B[1,2]
            U[1 * 4 + (2*b_A + 1)][col] = 1
            V[(2*b_A + 1) * 6 + (3*b_B + 2)][col] = 1
            W[(3*b_B + 2) * 2 + 1][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
