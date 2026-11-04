"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: partition 6x6 into 2x2 blocks of 3x3 matrices, use Strassen's
    7-product algorithm for the 2x2 block structure and schoolbook for the
    3x3 block multiplications.
    
    Total rank = 7 * 27 = 189.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # (36, 189)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (36, 189)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (36, 189)
    
    # Block structure: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    num_blocks = 2
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product (schoolbook, 27 scalar products)
    
    strassen_steps = [
        # (X_blocks, Y_blocks, C_contribution)
        # X_blocks: dict from (ib, jb) to coefficient for A[ib, jb]
        # Y_blocks: dict from (jb, kb) to coefficient for B[jb, kb]
        # C_contribution: dict from (ib, kb) to coefficient for C[ib, kb]
        
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'X': {(0,0): 1, (1,1): 1},
            'Y': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A10 + A11) * B00
        {
            'X': {(1,0): 1, (1,1): 1},
            'Y': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        # M3 = A00 * (B01 - B11)
        {
            'X': {(0,0): 1},
            'Y': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        # M4 = A11 * (B10 - B00)
        {
            'X': {(1,1): 1},
            'Y': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A00 + A01) * B11
        {
            'X': {(0,0): 1, (0,1): 1},
            'Y': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'X': {(1,0): 1, (0,0): -1},
            'Y': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'X': {(0,1): 1, (1,1): -1},
            'Y': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    col = 0
    
    # For each of the 7 Strassen multiplications, we do a full 3x3 matrix multiplication
    # Each 3x3 matrix multiplication has 27 terms: sum over r,s,c of X[r,s] * Y[s,c]
    
    for step in strassen_steps:
        X_coeffs = step['X']
        Y_coeffs = step['Y']
        C_coeffs = step['C']
        
        # For each of the 27 scalar products in the 3x3 matrix multiplication
        for r in range(block_size):  # row within block
            for s in range(block_size):  # sum index within block
                for c in range(block_size):  # col within block
                    
                    # U part: X[r, s] = sum_{ib,jb} X_coeffs[(ib,jb)] * A[ib*3+r, jb*3+s]
                    # U row index: (ib*3+r) * 6 + (jb*3+s)
                    
                    for (ib, jb), x_coeff in X_coeffs.items():
                        u_row = ib * block_size + r
                        u_col = jb * block_size + s
                        u_idx = u_row * n + u_col
                        U[u_idx, col] += x_coeff
                    
                    # V part: Y[s, c] = sum_{jb,kb} Y_coeffs[(jb,kb)] * B[jb*3+s, kb*3+c]
                    # V row index: (jb*3+s) * 6 + (kb*3+c)
                    
                    for (jb, kb), y_coeff in Y_coeffs.items():
                        v_row = jb * block_size + s
                        v_col = kb * block_size + c
                        v_idx = v_row * m + v_col
                        V[v_idx, col] += y_coeff
                    
                    # W part: The result M[r, c] contributes to C blocks
                    # C[ib, kb, r, c] += C_coeffs[(ib,kb)] * M[r,c]
                    # W row index (trace convention W[k*n+i, r]): 
                    # k = kb*3+c, i = ib*3+r
                    # W row index: (kb*3+c) * n + (ib*3+r)
                    
                    for (ib, kb), c_coeff in C_coeffs.items():
                        w_k = kb * block_size + c
                        w_i = ib * block_size + r
                        w_idx = w_k * n + w_i
                        W[w_idx, col] += c_coeff
                    
                    col += 1
        
        # Time check
        if time.time() - start_time > 130:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
