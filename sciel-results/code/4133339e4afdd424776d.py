import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We use the block decomposition:
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x6) = [[B11, B12, B13],
    #            [B21, B22, B23]] where each is 2x2
    # C (2x6) = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    # This requires 6 tensor products of 2x2x2.
    # The minimal rank of 2x2x2 matrix multiplication is 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping of local 2x2 indices to global indices
    # A: row i (0,1), col j (0,1,2,3)
    # B: row i (0,1,2,3), col j (0,1,2,3,4,5)
    # C: row i (0,1), col j (0,1,2,3,4,5) -> W index j*n + i
    
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(i, j):
        return i * p + j
        
    def w_idx(i, j):
        return j * n + i

    # Strassen matrix definitions for one 2x2x2 block
    # Each entry is (delta_a, delta_b, delta_c) for U, V, W
    # But we need to map local indices to global.
    
    # We iterate over the 6 blocks:
    # group (0 or 1) selects A half and B row half
    # block_col (0, 1, 2) selects B col half and C col half
    
    col_offset = 0
    for group in [0, 1]:
        # A rows 0,1 are fixed. A cols: group=0 -> 0,1; group=1 -> 2,3
        a_col_base = group * 2
        
        for block_col in [0, 1, 2]:
            # B rows: group=0 -> 0,1; group=1 -> 2,3
            b_row_base = group * 2
            # B cols: block_col=0 -> 0,1; block_col=1 -> 2,3; block_col=2 -> 4,5
            b_col_base = block_col * 2
            # C cols: block_col=0 -> 0,1; block_col=1 -> 2,3; block_col=2 -> 4,5
            c_col_base = block_col * 2
            
            # Define local indices for 2x2 A, B, C
            # A_local[i, j] -> A_global[i, a_col_base + j]
            # B_local[i, j] -> B_global[b_row_base + i, b_col_base + j]
            # C_local[i, j] -> C_global[i, c_col_base + j]
            
            # We will add 7 terms to U, V, W starting at col_offset
            
            # Term 1: (A00+A11)(B00+B11)
            # A: (0,0)+1, (1,1)+1
            # B: (0,0)+1, (1,1)+1
            # C: (0,0)+1, (1,1)+1
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 2: (A10+A11)B00
            # A: (1,0)+1, (1,1)+1
            # B: (0,0)+1
            # C: (1,0)+1, (1,1)-1
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] -= 1
            col_offset += 1
            
            # Term 3: A00(B01-B11)
            # A: (0,0)+1
            # B: (0,1)+1, (1,1)-1
            # C: (0,1)+1, (1,1)+1
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 4: A11(B10-B00)
            # A: (1,1)+1
            # B: (1,0)+1, (0,0)-1
            # C: (0,0)+1, (1,0)+1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
            # Term 5: (A00+A01)B11
            # A: (0,0)+1, (0,1)+1
            # B: (1,1)+1
            # C: (0,0)-1, (0,1)+1
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 6: (A10-A00)(B00+B01)
            # A: (1,0)+1, (0,0)-1
            # B: (0,0)+1, (0,1)+1
            # C: (1,1)+1
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 0), col_offset] -= 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 7: (A01-A11)(B10+B11)
            # A: (0,1)+1, (1,1)-1
            # B: (1,0)+1, (1,1)+1
            # C: (0,0)+1
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] -= 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
