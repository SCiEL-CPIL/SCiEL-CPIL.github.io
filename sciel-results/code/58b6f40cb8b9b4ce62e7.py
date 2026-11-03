import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use a block-Strassen decomposition.
    # Split A (2x4) into A_left (2x2) and A_right (2x2).
    # Split B (4x8) into B_top (2x8) and B_bottom (2x8).
    # C = A_left * B_top + A_right * B_bottom.
    # 
    # Each term is a 2x2 * 2x8 multiplication.
    # A 2x2 * 2x8 multiplication can be decomposed into 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication uses Strassen's algorithm with rank 7.
    # Total rank = 2 (terms) * 4 (blocks per term) * 7 (rank per block) = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = A * B
    #
    # M1 = (a0 + a3)(b0 + b3)
    # M2 = (a2 + a3)b0
    # M3 = a0(b1 - b3)
    # M4 = a3(b2 - b0)
    # M5 = (a0 + a1)b3
    # M6 = (a2 - a0)(b0 + b1)
    # M7 = (a1 - a3)(b2 + b3)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for A (U)
    # Index in A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coefficients for B (V)
    # Index in B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    v_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for C (W)
    # Index in C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    w_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # We have 8 blocks of 2x2x2 multiplication.
    # Block t (0..7) corresponds to:
    # A part: t < 4 -> A_left (cols 0,1), t >= 4 -> A_right (cols 2,3)
    # B part: t % 4 determines the column block of B (0, 1, 2, 3)
    #         t < 4 -> B_top (rows 0,1), t >= 4 -> B_bottom (rows 2,3)
    
    for t in range(8):
        # Determine offsets
        if t < 4:
            a_col_offset = 0
            b_row_offset = 0
        else:
            a_col_offset = 2
            b_row_offset = 2
            
        b_col_offset = (t % 4) * 2
        
        # Global indices for the 2x2 A block
        # A is 2x4. Flat index: i*m + col.
        # Local indices 0,1,2,3 map to:
        # 0: (0, a_col_offset)
        # 1: (0, a_col_offset+1)
        # 2: (1, a_col_offset)
        # 3: (1, a_col_offset+1)
        
        a_indices = [
            0 * m + a_col_offset,
            0 * m + a_col_offset + 1,
            1 * m + a_col_offset,
            1 * m + a_col_offset + 1
        ]
        
        # Global indices for the 2x2 B block
        # B is 4x8. Flat index: row*p + col.
        # Local indices 0,1,2,3 map to:
        # 0: (b_row_offset, b_col_offset)
        # 1: (b_row_offset, b_col_offset+1)
        # 2: (b_row_offset+1, b_col_offset)
        # 3: (b_row_offset+1, b_col_offset+1)
        
        b_indices = [
            b_row_offset * p + b_col_offset,
            b_row_offset * p + b_col_offset + 1,
            (b_row_offset + 1) * p + b_col_offset,
            (b_row_offset + 1) * p + b_col_offset + 1
        ]
        
        # Global indices for the 2x2 C block
        # C is 2x8. W index convention: k*n + i.
        # Local indices 0,1,2,3 map to:
        # 0: (0, b_col_offset) -> i=0, k=b_col_offset
        # 1: (0, b_col_offset+1) -> i=0, k=b_col_offset+1
        # 2: (1, b_col_offset) -> i=1, k=b_col_offset
        # 3: (1, b_col_offset+1) -> i=1, k=b_col_offset+1
        
        c_indices = [
            b_col_offset * n + 0,
            (b_col_offset + 1) * n + 0,
            b_col_offset * n + 1,
            (b_col_offset + 1) * n + 1
        ]
        
        # Fill the matrices for this block
        for l in range(7):
            r = t * 7 + l
            
            # U
            for local_idx, coeff in enumerate(u_coeffs[l]):
                if coeff != 0:
                    U[a_indices[local_idx], r] += coeff
            
            # V
            for local_idx, coeff in enumerate(v_coeffs[l]):
                if coeff != 0:
                    V[b_indices[local_idx], r] += coeff
            
            # W
            for local_idx, coeff in enumerate(w_coeffs[l]):
                if coeff != 0:
                    W[c_indices[local_idx], r] += coeff
    
    if time.time() > deadline:
        pass # Should not happen
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
