import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # We use a block-diagonal decomposition.
    # The multiplication C = A * B where A is 2x4, B is 4x8, C is 2x8.
    # We split the contraction dimension m=4 into two halves of size 2.
    # C = A_left * B_top + A_right * B_bottom
    # A_left is 2x2 (cols 0,1), B_top is 2x8 (rows 0,1)
    # A_right is 2x2 (cols 2,3), B_bottom is 2x8 (rows 2,3)
    
    # Each term (A_left * B_top) is a 2x2 * 2x8 multiplication.
    # We can further split B_top into four 2x2 blocks.
    # This results in 8 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Local indices for A, B, C are 0,1,2,3 corresponding to:
    # A: [[a00, a01], [a10, a11]] -> flat [a00, a01, a10, a11]
    # B: [[b00, b01], [b10, b11]] -> flat [b00, b01, b10, b11]
    # C: [[c00, c01], [c10, c11]] -> flat [c00, c01, c10, c11]
    
    # We define the 7 Strassen terms.
    # Each term is a tuple (U_local, V_local, W_local)
    # U_local, V_local, W_local are lists of (index, coeff)
    
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10 + a11)b00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a00(b01 - b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a11(b10 - b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a00 + a01)b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a10 - a00)(b00 + b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01 - a11)(b10 + b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    total_rank = 56
    
    # Initialize output matrices
    # U: 8 x 56
    # V: 32 x 56
    # W: 16 x 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 8 blocks
    # Block index t in 0..7
    # t = 0..3: A_left (cols 0,1) * B_top (rows 0,1, cols 2*t, 2*t+1)
    # t = 4..7: A_right (cols 2,3) * B_bottom (rows 2,3, cols 2*(t-4), 2*(t-4)+1)
    
    for t in range(8):
        base_r = t * 7
        
        # Determine A block columns
        if t < 4:
            a_col_offset = 0
        else:
            a_col_offset = 2
            
        # Determine B block rows
        if t < 4:
            b_row_offset = 0
        else:
            b_row_offset = 2
            
        # Determine B block columns
        b_col_block = t % 4
        b_col_offset = b_col_block * 2
        
        # Determine C block columns (same as B columns)
        c_col_offset = b_col_offset
        
        # Mapping local A indices (0,1,2,3) to global U indices
        # Local 0: (0,0) -> Global (0, a_col_offset) -> U[0*4 + a_col_offset]
        # Local 1: (0,1) -> Global (0, a_col_offset+1) -> U[0*4 + a_col_offset+1]
        # Local 2: (1,0) -> Global (1, a_col_offset) -> U[1*4 + a_col_offset]
        # Local 3: (1,1) -> Global (1, a_col_offset+1) -> U[1*4 + a_col_offset+1]
        a_map = [
            0 * m + a_col_offset,
            0 * m + a_col_offset + 1,
            1 * m + a_col_offset,
            1 * m + a_col_offset + 1
        ]
        
        # Mapping local B indices (0,1,2,3) to global V indices
        # Local 0: (0,0) -> Global (b_row_offset, b_col_offset) -> V[b_row_offset*p + b_col_offset]
        # Local 1: (0,1) -> Global (b_row_offset, b_col_offset+1) -> V[b_row_offset*p + b_col_offset+1]
        # Local 2: (1,0) -> Global (b_row_offset+1, b_col_offset) -> V[(b_row_offset+1)*p + b_col_offset]
        # Local 3: (1,1) -> Global (b_row_offset+1, b_col_offset+1) -> V[(b_row_offset+1)*p + b_col_offset+1]
        b_map = [
            b_row_offset * p + b_col_offset,
            b_row_offset * p + b_col_offset + 1,
            (b_row_offset + 1) * p + b_col_offset,
            (b_row_offset + 1) * p + b_col_offset + 1
        ]
        
        # Mapping local C indices (0,1,2,3) to global W indices
        # W convention: W[k*n + i, r]
        # Local 0: (0,0) -> i=0, k=c_col_offset -> W[c_col_offset*n + 0]
        # Local 1: (0,1) -> i=0, k=c_col_offset+1 -> W[(c_col_offset+1)*n + 0]
        # Local 2: (1,0) -> i=1, k=c_col_offset -> W[c_col_offset*n + 1]
        # Local 3: (1,1) -> i=1, k=c_col_offset+1 -> W[(c_col_offset+1)*n + 1]
        c_map = [
            c_col_offset * n + 0,
            (c_col_offset + 1) * n + 0,
            c_col_offset * n + 1,
            (c_col_offset + 1) * n + 1
        ]
        
        # Fill the matrices for this block
        for l in range(7):
            r = base_r + l
            u_terms, v_terms, w_terms = strassen_terms[l]
            
            for idx, coeff in u_terms:
                U[a_map[idx]][r] += coeff
                
            for idx, coeff in v_terms:
                V[b_map[idx]][r] += coeff
                
            for idx, coeff in w_terms:
                W[c_map[idx]][r] += coeff
                
    if time.time() > deadline:
        pass # Should not happen
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
