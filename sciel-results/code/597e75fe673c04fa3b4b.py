"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    """
    Returns a tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses Strassen for 2x2x2 sub-blocks and schoolbook for 2x1x2 sub-blocks.
    Rank is 3 * (7 + 4) = 33.
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    
    # We split B into 3 blocks of 3x2.
    # For each block, we compute A (2x3) * B_block (3x2) = C_block (2x2).
    # A is split into A_L (2x2) and A_R (2x1).
    # B_block is split into B_T (2x2) and B_B (1x2).
    # C_block = A_L B_T + A_R B_B.
    # Rank(A_L B_T) <= 7 (Strassen).
    # Rank(A_R B_B) = 4 (Schoolbook).
    # Total rank per block = 11.
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col_offset = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col_offset
        if col_offset >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, col_offset] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, col_offset] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, col_offset] += val
        col_offset += 1
        
    # Strassen terms for 2x2x2
    # Local indices: i,j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We process 3 blocks of B (cols 0-1, 2-3, 4-5)
    for block in range(3):
        # B col offset
        k_offset = block * 2
        
        # --- Strassen Part (A_L 2x2 * B_T 2x2) ---
        # A_L uses A cols 0,1. Rows 0,1.
        # B_T uses B rows 0,1. Cols k_offset, k_offset+1.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, k=k_offset. B11: j=1, k=k_offset+1.
        # W: C00 (k=0, i=0), C11 (k=1, i=1).
        # Note: W index is k*n + i.
        # Local k for C is 0,1. Global k is k_offset, k_offset+1.
        
        u_idx = [0*m + 0, 1*m + 1] # A00, A11
        u_val = [1, 1]
        v_idx = [0*p + k_offset, 1*p + (k_offset+1)] # B00, B11
        v_val = [1, 1]
        w_idx = [0*n + 0, 1*n + 1] # C00, C11
        w_val = [1, 1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M2 = (A10 + A11) * B00
        # A10: i=1, j=0. A11: i=1, j=1.
        # B00: j=0, k=k_offset.
        # W: C10 (k=0, i=1).
        u_idx = [1*m + 0, 1*m + 1]
        u_val = [1, 1]
        v_idx = [0*p + k_offset]
        v_val = [1]
        w_idx = [0*n + 1]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M3 = A00 * (B01 - B11)
        # A00: i=0, j=0.
        # B01: j=0, k=k_offset+1. B11: j=1, k=k_offset+1.
        # W: C01 (k=1, i=0).
        u_idx = [0*m + 0]
        u_val = [1]
        v_idx = [0*p + (k_offset+1), 1*p + (k_offset+1)]
        v_val = [1, -1]
        w_idx = [1*n + 0]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M4 = A11 * (B10 - B00)
        # A11: i=1, j=1.
        # B10: j=1, k=k_offset. B00: j=0, k=k_offset.
        # W: C10 (k=0, i=1).
        u_idx = [1*m + 1]
        u_val = [1]
        v_idx = [1*p + k_offset, 0*p + k_offset]
        v_val = [1, -1]
        w_idx = [0*n + 1]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M5 = (A00 + A01) * B11
        # A00: i=0, j=0. A01: i=0, j=1.
        # B11: j=1, k=k_offset+1.
        # W: C01 (k=1, i=0).
        u_idx = [0*m + 0, 0*m + 1]
        u_val = [1, 1]
        v_idx = [1*p + (k_offset+1)]
        v_val = [1]
        w_idx = [1*n + 0]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A10: i=1, j=0. A00: i=0, j=0.
        # B00: j=0, k=k_offset. B01: j=0, k=k_offset+1.
        # W: C11 (k=1, i=1).
        u_idx = [1*m + 0, 0*m + 0]
        u_val = [1, -1]
        v_idx = [0*p + k_offset, 0*p + (k_offset+1)]
        v_val = [1, 1]
        w_idx = [1*n + 1]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # M7 = (A01 - A11) * (B10 + B11)
        # A01: i=0, j=1. A11: i=1, j=1.
        # B10: j=1, k=k_offset. B11: j=1, k=k_offset+1.
        # W: C00 (k=0, i=0).
        u_idx = [0*m + 1, 1*m + 1]
        u_val = [1, -1]
        v_idx = [1*p + k_offset, 1*p + (k_offset+1)]
        v_val = [1, 1]
        w_idx = [0*n + 0]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # --- Schoolbook Part (A_R 2x1 * B_B 1x2) ---
        # A_R uses A col 2. Rows 0,1.
        # B_B uses B row 2. Cols k_offset, k_offset+1.
        
        # C00 += A02 * B20
        u_idx = [0*m + 2]
        u_val = [1]
        v_idx = [2*p + k_offset]
        v_val = [1]
        w_idx = [0*n + 0]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # C01 += A02 * B21
        u_idx = [0*m + 2]
        u_val = [1]
        v_idx = [2*p + (k_offset+1)]
        v_val = [1]
        w_idx = [1*n + 0]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # C10 += A12 * B20
        u_idx = [1*m + 2]
        u_val = [1]
        v_idx = [2*p + k_offset]
        v_val = [1]
        w_idx = [0*n + 1]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)
        
        # C11 += A12 * B21
        u_idx = [1*m + 2]
        u_val = [1]
        v_idx = [2*p + (k_offset+1)]
        v_val = [1]
        w_idx = [1*n + 1]
        w_val = [1]
        add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
