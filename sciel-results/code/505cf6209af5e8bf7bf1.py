"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    
    # Define the rank-7 Strassen decomposition for 2x2x2
    # U_core: 4x7, V_core: 4x7, W_core: 4x7
    # Indices for U: i*m + j
    # Indices for V: j*p_local + k_local (p_local=2) -> j*2 + k
    # Indices for W: k*n + i -> k*2 + i
    
    U_core = np.zeros((4, 7))
    V_core = np.zeros((4, 7))
    W_core = np.zeros((4, 7))
    
    # Strassen factors (M1..M7)
    # M1: (a00+a11)(b00+b11)
    U_core[0, 0] += 1; U_core[3, 0] += 1  # a00 (0), a11 (3)
    V_core[0, 0] += 1; V_core[3, 0] += 1  # b00 (0), b11 (3)
    W_core[0, 0] += 1; W_core[3, 0] += 1  # c00 (0), c11 (3)
    
    # M2: (a10+a11)b00
    U_core[2, 1] += 1; U_core[3, 1] += 1  # a10 (2), a11 (3)
    V_core[0, 1] += 1                     # b00 (0)
    W_core[2, 1] += 1; W_core[3, 1] -= 1  # c10 (2), c11 (3)
    
    # M3: a00(b01-b11)
    U_core[0, 2] += 1                     # a00 (0)
    V_core[1, 2] += 1; V_core[3, 2] -= 1  # b01 (1), b11 (3)
    W_core[1, 2] += 1; W_core[3, 2] += 1  # c01 (1), c11 (3)
    
    # M4: a11(b10-b00)
    U_core[3, 3] += 1                     # a11 (3)
    V_core[2, 3] += 1; V_core[0, 3] -= 1  # b10 (2), b00 (0)
    W_core[0, 3] += 1; W_core[2, 3] += 1  # c00 (0), c10 (2)
    
    # M5: (a00+a01)b11
    U_core[0, 4] += 1; U_core[1, 4] += 1  # a00 (0), a01 (1)
    V_core[3, 4] += 1                     # b11 (3)
    W_core[0, 4] -= 1; W_core[1, 4] += 1  # c00 (0), c01 (1)
    
    # M6: (a10-a00)(b00+b01)
    U_core[2, 5] += 1; U_core[0, 5] -= 1  # a10 (2), a00 (0)
    V_core[0, 5] += 1; V_core[1, 5] += 1  # b00 (0), b01 (1)
    W_core[3, 5] += 1                     # c11 (3)
    
    # M7: (a01-a11)(b10+b11)
    U_core[1, 6] += 1; U_core[3, 6] -= 1  # a01 (1), a11 (3)
    V_core[2, 6] += 1; V_core[3, 6] += 1  # b10 (2), b11 (3)
    W_core[0, 6] += 1                     # c00 (0)
    
    # Construct full factors using the block structure
    # U is 4 x 56. It consists of 8 copies of U_core.
    # V is 32 x 56. Row index j*16 + k.
    # For block t, k = 2t + k'. Row = j*16 + 2t + k'.
    # This is not a simple horizontal repeat of V_core because the row indices shift.
    # However, we can build V by placing V_core into the appropriate rows for each block t.
    
    U = np.tile(U_core, (1, num_blocks))
    
    V = np.zeros((m * p, 7 * num_blocks))
    W = np.zeros((p * n, 7 * num_blocks))
    
    for t in range(num_blocks):
        col_start = t * 7
        col_end = (t + 1) * 7
        
        # V rows for this block: j in 0..1, k' in 0..1
        # Global row index: j * p + (2*t + k')
        for j in range(m):
            for k_prime in range(2):
                global_row = j * p + (2 * t + k_prime)
                core_row = j * 2 + k_prime
                V[global_row, col_start:col_end] = V_core[core_row, :]
        
        # W rows for this block: k' in 0..1, i in 0..1
        # Global row index: (2*t + k') * n + i
        for k_prime in range(2):
            for i in range(n):
                global_row = (2 * t + k_prime) * n + i
                core_row = k_prime * 2 + i
                W[global_row, col_start:col_end] = W_core[core_row, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
