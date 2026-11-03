import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # n=2, m=4, p=8
    n, m, p = 2, 4, 8
    
    # Strassen 2x2 * 2x2 factors (rank 7)
    # U: 4x7 (A factors)
    # V: 4x7 (B factors)
    # W: 4x7 (C factors)
    
    # Standard Strassen decomposition for C = A*B
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float) # 8 x 56
    V = np.zeros((m * p, rank), dtype=float) # 32 x 56
    W = np.zeros((p * n, rank), dtype=float) # 16 x 56
    
    # The multiplication is C = A * B
    # A is 2x4, B is 4x8
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    
    # Block 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B)
    # A1 indices in A: i*4 + j for i in 0,1 and j in 0,1 -> 0,1,4,5
    # B1 indices in B: j*8 + k for j in 0,1 and k in 0..7 -> 0..7, 8..15
    # C indices in C: k*n + i for k in 0..7 and i in 0,1 -> 0,1, 2,3, ...
    
    # We split the 7 Strassen products into 4 groups of ?
    # No, for 2x2 * 2x8, we use 28 products.
    # The 2x8 B is split into 4 2x2 blocks B1, B2, B3, B4 (cols 0-1, 2-3, 4-5, 6-7).
    # For each block r (0..3), we use the 7 Strassen products.
    # Total 28 products for the first half (A1*B1).
    
    # Mapping for A1:
    # Local A index la (0..3) maps to global A index:
    # la=0 (a00) -> 0
    # la=1 (a01) -> 1
    # la=2 (a10) -> 4
    # la=3 (a11) -> 5
    a1_map = [0, 1, 4, 5]
    
    # Mapping for B1 (rows 0,1 of B, all 8 cols):
    # Local B index lb (0..3) corresponds to (row, col_block)?
    # Strassen V_local has 4 rows (b00, b01, b10, b11).
    # These correspond to the 4 entries of the 2x2 B block.
    # For block r (cols 2r, 2r+1), the global B indices are:
    # b00 -> 0*8 + 2r
    # b01 -> 0*8 + 2r + 1
    # b10 -> 1*8 + 2r
    # b11 -> 1*8 + 2r + 1
    
    # Mapping for C:
    # Local C index lc (0..3) corresponds to (row, col_block)?
    # Strassen W_local has 4 rows (c00, c01, c10, c11).
    # For block r (cols 2r, 2r+1), the global C indices are:
    # c00 -> (2r)*2 + 0
    # c01 -> (2r+1)*2 + 0
    # c10 -> (2r)*2 + 1
    # c11 -> (2r+1)*2 + 1
    
    # Construct Block 1 (A1 * B1) -> Rank 28 (cols 0..27 of U,V,W)
    for r in range(4): # 4 blocks of 2x2 in B
        base_r = r * 7
        for l in range(7): # 7 Strassen products
            r_idx = base_r + l
            
            # U
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a1_map[la], r_idx] += U_local[la, l]
            
            # V
            b00_idx = 0 * 8 + 2 * r
            b01_idx = 0 * 8 + 2 * r + 1
            b10_idx = 1 * 8 + 2 * r
            b11_idx = 1 * 8 + 2 * r + 1
            b_map = [b00_idx, b01_idx, b10_idx, b11_idx]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b_map[lb], r_idx] += V_local[lb, l]
            
            # W
            c00_idx = (2 * r) * 2 + 0
            c01_idx = (2 * r + 1) * 2 + 0
            c10_idx = (2 * r) * 2 + 1
            c11_idx = (2 * r + 1) * 2 + 1
            c_map = [c00_idx, c01_idx, c10_idx, c11_idx]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r_idx] += W_local[lc, l]
                    
    # Construct Block 2 (A2 * B2) -> Rank 28 (cols 28..55 of U,V,W)
    # A2 indices: i*4 + (2+j) for i in 0,1 and j in 0,1 -> 2,3,6,7
    a2_map = [2, 3, 6, 7]
    
    # B2 indices: rows 2,3 of B
    # b00 -> 2*8 + 2r
    # b01 -> 2*8 + 2r + 1
    # b10 -> 3*8 + 2r
    # b11 -> 3*8 + 2r + 1
    
    for r in range(4):
        base_r = 28 + r * 7
        for l in range(7):
            r_idx = base_r + l
            
            # U
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a2_map[la], r_idx] += U_local[la, l]
            
            # V
            b00_idx = 2 * 8 + 2 * r
            b01_idx = 2 * 8 + 2 * r + 1
            b10_idx = 3 * 8 + 2 * r
            b11_idx = 3 * 8 + 2 * r + 1
            b_map = [b00_idx, b01_idx, b10_idx, b11_idx]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b_map[lb], r_idx] += V_local[lb, l]
            
            # W
            c00_idx = (2 * r) * 2 + 0
            c01_idx = (2 * r + 1) * 2 + 0
            c10_idx = (2 * r) * 2 + 1
            c11_idx = (2 * r + 1) * 2 + 1
            c_map = [c00_idx, c01_idx, c10_idx, c11_idx]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r_idx] += W_local[lc, l]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
