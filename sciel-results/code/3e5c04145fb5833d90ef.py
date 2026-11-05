import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for 6x6 matrix multiplication.
    Strategy:
    1. Block the 6x6 matrices into a 3x3 grid of 2x2 sub-matrices.
    2. Perform block matrix multiplication: 27 block products.
    3. Each block product (2x2 * 2x2) is computed using Strassen's algorithm (Rank 7).
    4. Total Rank = 27 * 7 = 189.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Parameters for the block structure
    block_count = 3      # 3x3 blocks
    block_size = 2       # 2x2 matrices
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U coefficients for A: [(index, coeff), ...]
    # V coefficients for B: [(index, coeff), ...]
    # W coefficients for C: [(index, coeff), ...]
    
    # Strassen terms definition
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00      -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)      -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)      -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11      -> C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01)-> C11 += M6
    # M7 = (A01-A11)(B10+B11)-> C00 += M7
    
    strassen_u = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, 1)],   # M2
        [(0, 1)],           # M3
        [(3, 1)],           # M4
        [(0, 1), (1, 1)],   # M5
        [(2, 1), (0, -1)],  # M6
        [(1, 1), (3, -1)]   # M7
    ]
    
    strassen_v = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(1, 1), (3, -1)],  # M3
        [(2, 1), (0, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)]    # M7
    ]
    
    strassen_w = [
        [(0, 1), (3, 1)],   # M1 -> C00, C11
        [(2, 1), (3, -1)],  # M2 -> C10, C11
        [(1, 1), (3, 1)],   # M3 -> C01, C11
        [(0, 1), (2, 1)],   # M4 -> C00, C10
        [(1, 1), (0, -1)],  # M5 -> C01, C00
        [(3, 1)],           # M6 -> C11
        [(0, 1)]            # M7 -> C00
    ]

    rank = 189
    dim_full = n * m # 36
    
    # Initialize factor matrices
    U = np.zeros((dim_full, rank), dtype=int)
    V = np.zeros((dim_full, rank), dtype=int)
    W = np.zeros((dim_full, rank), dtype=int)
    
    # Generate decomposition
    col_idx = 0
    for i_block in range(block_count):
        for k_block in range(block_count):
            for j_block in range(block_count):
                for t_idx in range(7):
                    # U: A[2*i_block + r, 2*k_block + c]
                    for a_idx, a_coeff in strassen_u[t_idx]:
                        a_r, a_c = divmod(a_idx, block_size)
                        r_full = block_size * i_block + a_r
                        c_full = block_size * k_block + a_c
                        U[r_full * n + c_full, col_idx] += a_coeff
                    
                    # V: B[2*k_block + r, 2*j_block + c]
                    for b_idx, b_coeff in strassen_v[t_idx]:
                        b_r, b_c = divmod(b_idx, block_size)
                        r_full = block_size * k_block + b_r
                        c_full = block_size * j_block + b_c
                        V[r_full * m + c_full, col_idx] += b_coeff
                    
                    # W: C[2*j_block + r, 2*i_block + c]
                    # Note: Output C indices are (row, col) corresponding to (j_block, i_block) block structure
                    # C row block is j_block, col block is i_block
                    for c_idx_out, c_coeff in strassen_w[t_idx]:
                        c_r, c_c = divmod(c_idx_out, block_size)
                        r_full = block_size * j_block + c_r
                        c_full = block_size * i_block + c_c
                        W[r_full * p + c_full, col_idx] += c_coeff
                    
                    col_idx += 1
                    
    elapsed = time.time() - t0
    if elapsed > 145:
        # Safety check
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
