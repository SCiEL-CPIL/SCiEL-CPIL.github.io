import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # A is n x m (2x2)
    # B is m x p (2x12)
    # C is n x p (2x12)
    
    # We decompose B into 6 blocks of size 2x2.
    # num_blocks = p // 2 = 6
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Initialize U, V, W
    # U shape: (n*m, total_rank) = (4, 42)
    # V shape: (m*p, total_rank) = (24, 42)
    # W shape: (p*n, total_rank) = (24, 42)
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Define Strassen tensors for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Linear forms for A (U0)
    # S0 = a00 + a11
    # S1 = a10 + a11
    # S2 = a00
    # S3 = a11
    # S4 = a00 + a01
    # S5 = a10 - a00
    # S6 = a01 - a11
    
    # U0 maps A entries (flattened: a00, a01, a10, a11) to forms
    U0 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    # Linear forms for B (V0)
    # T0 = b00 + b11
    # T1 = b00
    # T2 = b01 - b11
    # T3 = b10 - b00
    # T4 = b11
    # T5 = b00 + b01
    # T6 = b10 + b11
    
    # V0 maps B entries (flattened: b00, b01, b10, b11) to forms
    V0 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    # Output coefficients (W0)
    # M0 = S0 * T0 = (a00+a11)(b00+b11)
    # M1 = S1 * T1 = (a10+a11)b00
    # M2 = S2 * T2 = a00(b01-b11)
    # M3 = S3 * T3 = a11(b10-b00)
    # M4 = S4 * T4 = (a00+a01)b11
    # M5 = S5 * T5 = (a10-a00)(b00+b01)
    # M6 = S6 * T6 = (a01-a11)(b10+b11)
    
    # c00 = M0 + M3 - M4 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M5
    
    # W0 maps forms to C entries (flattened: c00, c01, c10, c11)
    W0 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Fill U
    # U[i, 7*j + t] = U0[i, t]
    # This is equivalent to tiling U0 along the columns
    U = np.tile(U0, (1, num_blocks))
    
    # Fill V
    # For each block j:
    # B block entries:
    # b00 -> B[0, 2j] -> V row 2j
    # b01 -> B[0, 2j+1] -> V row 2j+1
    # b10 -> B[1, 2j] -> V row m*p//2 + 2j  (since B is 2x12, row 1 starts at index 12)
    # b11 -> B[1, 2j+1] -> V row m*p//2 + 2j + 1
    
    # V0 rows: 0:b00, 1:b01, 2:b10, 3:b11
    # V rows for block j:
    row_b00 = 0 * p + 2 * j
    row_b01 = 0 * p + 2 * j + 1
    row_b10 = 1 * p + 2 * j
    row_b11 = 1 * p + 2 * j + 1
    
    # V columns for block j: 7*j to 7*j+6
    col_start = 7 * j
    col_end = col_start + 7
    
    for j in range(num_blocks):
        V[row_b00, col_start:col_end] = V0[0, :]
        V[row_b01, col_start:col_end] = V0[1, :]
        V[row_b10, col_start:col_end] = V0[2, :]
        V[row_b11, col_start:col_end] = V0[3, :]
        
    # Fill W
    # For each block j:
    # C block entries:
    # c00 -> C[0, 2j] -> W index (2j)*n + 0 = 4j
    # c01 -> C[0, 2j+1] -> W index (2j+1)*n + 0 = 4j+2
    # c10 -> C[1, 2j] -> W index (2j)*n + 1 = 4j+1
    # c11 -> C[1, 2j+1] -> W index (2j+1)*n + 1 = 4j+3
    
    # W0 rows: 0:c00, 1:c01, 2:c10, 3:c11
    
    for j in range(num_blocks):
        row_c00 = (2 * j) * n + 0
        row_c01 = (2 * j + 1) * n + 0
        row_c10 = (2 * j) * n + 1
        row_c11 = (2 * j + 1) * n + 1
        
        col_start = 7 * j
        col_end = col_start + 7
        
        W[row_c00, col_start:col_end] = W0[0, :]
        W[row_c01, col_start:col_end] = W0[1, :]
        W[row_c10, col_start:col_end] = W0[2, :]
        W[row_c11, col_start:col_end] = W0[3, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
