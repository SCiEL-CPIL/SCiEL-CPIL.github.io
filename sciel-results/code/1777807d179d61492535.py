"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    def get_block_entries(bi, bj):
        r0 = 2 * bi
        c0 = 2 * bj
        return [r0*4 + c0, r0*4 + c0+1, (r0+1)*4 + c0, (r0+1)*4 + c0+1]
    
    # Strassen 2x2: X=[[x0,x1],[x2,x3]], Y=[[y0,y1],[y2,y3]]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # s1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # s2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # s3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # s4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # s5
        ([0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # s6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # s7
    ]
    
    # Block-level Strassen coefficients for blocks [11, 12, 21, 22]
    block_X_coeffs = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [0, 0, 1, -1],  # M6: A21-A11
        [0, 1, 0, -1],  # M7: A12-A22
    ]
    block_Y_coeffs = [
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12-B22
        [0, 0, 1, -1],  # M4: B21-B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1],   # M7: B21+B22
    ]
    block_outputs = [
        [1, 0, 0, 1],   # M1 -> C11, C22
        [0, 0, 1, -1],  # M2 -> C21, -C22
        [0, 1, 0, 1],   # M3 -> C12, C22
        [1, 0, 1, 0],   # M4 -> C11, C21
        [-1, 1, 0, 0],  # M5 -> -C11, C12
        [0, 0, 0, 1],   # M6 -> C22
        [1, 0, 0, 0],   # M7 -> C11
    ]
    
    col = 0
    for b in range(7):
        for s in range(7):
            x_coeff_block, y_coeff_block, z_coeff_block = strassen_terms[s]
            
            U_row = np.zeros(16)
            for bp in range(4):
                bc = block_X_coeffs[b][bp]
                if bc == 0:
                    continue
                entries = get_block_entries(bp // 2, bp % 2)
                for li in range(4):
                    if x_coeff_block[li] != 0:
                        U_row[entries[li]] += bc * x_coeff_block[li]
            
            V_row = np.zeros(16)
            for bp in range(4):
                bc = block_Y_coeffs[b][bp]
                if bc == 0:
                    continue
                entries = get_block_entries(bp // 2, bp % 2)
                for li in range(4):
                    if y_coeff_block[li] != 0:
                        V_row[entries[li]] += bc * y_coeff_block[li]
            
            W_row = np.zeros(16)
            for bp in range(4):
                bc = block_outputs[b][bp]
                if bc == 0:
                    continue
                entries = get_block_entries(bp // 2, bp % 2)
                for li in range(4):
                    if z_coeff_block[li] != 0:
                        W_row[entries[li]] += bc * z_coeff_block[li]
            
            U[:, col] = U_row
            V[:, col] = V_row
            W[:, col] = W_row
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
