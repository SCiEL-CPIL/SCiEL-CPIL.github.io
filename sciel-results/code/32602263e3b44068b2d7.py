import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    # Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B (4x8) into B1 (2x8, rows 0-1) and B2 (2x8, rows 2-3)
    # C = A1 B1 + A2 B2
    
    # Each 2x2x8 multiplication is 4 independent 2x2x2 multiplications
    # Each 2x2x2 uses Strassen (rank 7)
    # Total rank = 2 * 4 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # j_offset: 0 for A1/B1, 2 for A2/B2
        # k_offset: 0, 2, 4, 6 for the 4 blocks of 2 cols each in B
        
        # Local indices for 2x2 A block: i in 0..1, j_loc in 0..1
        # Global j = j_offset + j_loc
        # Local indices for 2x2 B block: j_loc in 0..1, k_loc in 0..1
        # Global k = k_offset + k_loc
        
        # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[n*0 + (j_offset+0)][col] = 1
        U[n*1 + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1   # z00
        W[(k_offset+1)*n + 0][col] = 0   # z01 (not in M1)
        W[(k_offset+0)*n + 1][col] = 0   # z10 (not in M1)
        W[(k_offset+1)*n + 1][col] = 1   # z11
        col += 1
        
        # M2: (A[1,0] + A[1,1]) * B[0,0]
        U[n*1 + (j_offset+0)][col] = 1
        U[n*1 + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + 1][col] = 1   # z10
        W[(k_offset+1)*n + 1][col] = -1  # z11
        col += 1
        
        # M3: A[0,0] * (B[0,1] - B[1,1])
        U[n*0 + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + 0][col] = 1   # z01
        W[(k_offset+1)*n + 1][col] = 1   # z11
        col += 1
        
        # M4: A[1,1] * (B[1,0] - B[0,0])
        U[n*1 + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + 0][col] = 1   # z00
        W[(k_offset+0)*n + 1][col] = 1   # z10
        col += 1
        
        # M5: (A[0,0] + A[0,1]) * B[1,1]
        U[n*0 + (j_offset+0)][col] = 1
        U[n*0 + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = -1  # z00
        W[(k_offset+1)*n + 0][col] = 1   # z01
        col += 1
        
        # M6: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[n*1 + (j_offset+0)][col] = 1
        U[n*0 + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + 1][col] = 1   # z11
        col += 1
        
        # M7: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[n*0 + (j_offset+1)][col] = 1
        U[n*1 + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1   # z00
        col += 1

    # Part 1: A1 (j_offset=0) * B1 (rows 0-1)
    for k_off in [0, 2, 4, 6]:
        add_strassen_block(0, k_off)
        
    # Part 2: A2 (j_offset=2) * B2 (rows 2-3)
    for k_off in [0, 2, 4, 6]:
        add_strassen_block(2, k_off)
        
    return {"U": U, "V": V, "W": W}
