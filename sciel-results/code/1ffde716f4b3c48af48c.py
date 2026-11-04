"""Rank-minimal tensor decomposition for 2x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    
    # We construct a rank-56 decomposition.
    # Strategy:
    # C = A * B
    # A is 2x4, B is 4x8, C is 2x8.
    #
    # Partition B into 4 blocks of 4x2: B = [B0, B1, B2, B3]
    # C = [C0, C1, C2, C3] where Ck = A * Bk (2x2)
    #
    # For each Ck, partition A into [A1, A2] (2x2 each) and Bk into [Bk1, Bk2] (2x2 each).
    # Ck = A1 * Bk1 + A2 * Bk2
    #
    # We use Strassen's algorithm for the 2x2 * 2x2 multiplications.
    # Strassen computes X * Y (2x2) with 7 products:
    # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y = [[z00, z01], [z10, z11]]
    #
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y10 - y11)
    # M4 = x11 * (y01 - y11)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    #
    # z00 = M1 + M3 - M4 + M6
    # z01 = M2 + M5 - M6 + M7
    # z10 = M2 + M3 + M5
    # z11 = M1 - M2 + M4 + M7
    #
    # Total rank = 4 (B blocks) * 2 (A sub-blocks) * 7 (Strassen terms) = 56.

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx: index in A (i*m + j)
    # v_idx: index in B (j*p + k)
    # w_idx: index in C (k*n + i)
    # coeff: scalar
    
    def add_term(u_indices, v_indices, w_indices, coeff=1.0):
        nonlocal col
        if col >= rank:
            return
        for u in u_indices:
            U[u, col] += coeff
        for v in v_indices:
            V[v, col] += coeff
        for w in w_indices:
            W[w, col] += coeff
        col += 1

    # Iterate over the 4 blocks of B (each 4x2)
    for block in range(4):
        # B block columns are [block*2, block*2+1]
        b_col_start = block * 2
        
        # Iterate over the 2 sub-blocks of A (and corresponding B rows)
        # Sub 0: A rows 0-1, cols 0-1 (A1), B rows 0-1 (Bk1)
        # Sub 1: A rows 0-1, cols 2-3 (A2), B rows 2-3 (Bk2)
        
        for sub in range(2):
            # A sub-block rows: 0, 1
            # A sub-block cols: sub*2, sub*2+1
            a_col_start = sub * 2
            
            # B sub-block rows: sub*2, sub*2+1
            # B sub-block cols: block*2, block*2+1
            b_row_start = sub * 2
            
            # C sub-block rows: 0, 1
            # C sub-block cols: block*2, block*2+1
            c_col_start = block * 2
            
            # Define indices for A sub-block (2x2)
            # A[0, a_col_start], A[0, a_col_start+1], A[1, a_col_start], A[1, a_col_start+1]
            a00_idx = 0 * m + a_col_start
            a01_idx = 0 * m + a_col_start + 1
            a10_idx = 1 * m + a_col_start
            a11_idx = 1 * m + a_col_start + 1
            
            # Define indices for B sub-block (2x2)
            # B[b_row_start, c_col_start], B[b_row_start, c_col_start+1], 
            # B[b_row_start+1, c_col_start], B[b_row_start+1, c_col_start+1]
            b00_idx = b_row_start * p + c_col_start
            b01_idx = b_row_start * p + c_col_start + 1
            b10_idx = (b_row_start + 1) * p + c_col_start
            b11_idx = (b_row_start + 1) * p + c_col_start + 1
            
            # Define indices for C sub-block (2x2)
            # C[0, c_col_start], C[0, c_col_start+1], C[1, c_col_start], C[1, c_col_start+1]
            c00_idx = c_col_start * n + 0
            c01_idx = c_col_start * n + 1
            c10_idx = (c_col_start + 1) * n + 0
            c11_idx = (c_col_start + 1) * n + 1

            # Strassen Term 1: (A00 + A11) * (B00 + B11)
            # U: A00, A11
            # V: B00, B11
            # W: C00, C01, C10, C11
            # z00 += M1, z01 += M1, z10 += M1, z11 += M1
            add_term([a00_idx, a11_idx], [b00_idx, b11_idx], [c00_idx, c01_idx, c10_idx, c11_idx])
            
            # Strassen Term 2: (A10 + A11) * B00
            # U: A10, A11
            # V: B00
            # W: C01, C10
            # z01 += M2, z10 += M2
            add_term([a10_idx, a11_idx], [b00_idx], [c01_idx, c10_idx])
            
            # Strassen Term 3: A00 * (B10 - B11)
            # U: A00
            # V: B10, -B11
            # W: C00, -C01
            # z00 += M3, z01 -= M3
            add_term([a00_idx], [b10_idx, -b11_idx], [c00_idx, -c01_idx])
            
            # Strassen Term 4: A11 * (B01 - B11)
            # U: A11
            # V: B01, -B11
            # W: -C00, C11
            # z00 -= M4, z11 += M4
            add_term([a11_idx], [b01_idx, -b11_idx], [-c00_idx, c11_idx])
            
            # Strassen Term 5: (A00 + A01) * B11
            # U: A00, A01
            # V: B11
            # W: C01, C10
            # z01 += M5, z10 += M5
            add_term([a00_idx, a01_idx], [b11_idx], [c01_idx, c10_idx])
            
            # Strassen Term 6: (A10 - A00) * (B00 + B01)
            # U: A10, -A00
            # V: B00, B01
            # W: C00, C10
            # z00 += M6, z10 += M6
            add_term([a10_idx, -a00_idx], [b00_idx, b01_idx], [c00_idx, c10_idx])
            
            # Strassen Term 7: (A01 - A11) * (B10 + B11)
            # U: A01, -A11
            # V: B10, B11
            # W: -C01, C11
            # z01 -= M7, z11 += M7
            add_term([a01_idx, -a11_idx], [b10_idx, b11_idx], [-c01_idx, c11_idx])

    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
