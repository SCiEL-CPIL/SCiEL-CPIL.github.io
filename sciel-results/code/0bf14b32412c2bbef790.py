import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 3x3 grid of 2x2 blocks
    # Each 2x2 multiplication uses Strassen (rank 7)
    # Total rank = 3 * 3 * 7 = 63
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_size = 2
    num_blocks = 3 # 6 / 2
    
    # Strassen 2x2 definitions
    # Each entry is a list of (local_row, local_col, coeff)
    # U: A indices, V: B indices, W: C indices (k, i)
    
    # M1 = (A00 + A11)(B00 + B11)
    A_0 = [(0,0,1), (1,1,1)]
    B_0 = [(0,0,1), (1,1,1)]
    C_0 = [(0,0,1), (1,1,1)] # C00 += 1, C11 += 1
    
    # M2 = (A10 + A11)B00
    A_1 = [(1,0,1), (1,1,1)]
    B_1 = [(0,0,1)]
    C_1 = [(0,1,1)] # C10 += 1 (Wait, C10 is row 1, col 0. W index k=0, i=1)
    # Let's verify C indices for Strassen 2x2
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M2:
    # C10 += 1.
    # W index for C10: k=0, i=1.
    # W[k*n + i] -> W[0*6 + 1] = W[1].
    # In my loop, I will use local C indices (i_c, k_c) where C[i_c, k_c] is the element.
    # Then map to W[k_c * n + i_c].
    
    # Let's define C contributions as (i_c, k_c, coeff)
    C_1 = [(1,0,1)]
    
    # M3 = A00(B01 - B11)
    A_2 = [(0,0,1)]
    B_2 = [(0,1,1), (1,1,-1)]
    C_2 = [(0,1,1)] # C01 += 1
    
    # M4 = A11(B10 - B00)
    A_3 = [(1,1,1)]
    B_3 = [(1,0,1), (0,0,-1)]
    C_3 = [(0,0,1), (1,0,1)] # C00 += 1, C10 += 1
    
    # M5 = (A00 + A01)B11
    A_4 = [(0,0,1), (0,1,1)]
    B_4 = [(1,1,1)]
    C_4 = [(0,0,-1), (0,1,1)] # C00 -= 1, C01 += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    A_5 = [(1,0,1), (0,0,-1)]
    B_5 = [(0,0,1), (0,1,1)]
    C_5 = [(1,1,1)] # C11 += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    A_6 = [(0,1,1), (1,1,-1)]
    B_6 = [(1,0,1), (1,1,1)]
    C_6 = [(0,0,1), (1,1,-1)] # C00 += 1, C11 -= 1
    
    A_combs = [A_0, A_1, A_2, A_3, A_4, A_5, A_6]
    B_combs = [B_0, B_1, B_2, B_3, B_4, B_5, B_6]
    C_combs = [C_0, C_1, C_2, C_3, C_4, C_5, C_6]
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for t in range(7):
                    # Update U
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = I * block_size + rb
                        j_glob = K * block_size + cb
                        U[i_glob * m + j_glob][col] += coeff
                        
                    # Update V
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = K * block_size + rb
                        k_glob = J * block_size + cb
                        V[j_glob * p + k_glob][col] += coeff
                        
                    # Update W
                    for rb, cb, coeff in C_combs[t]:
                        # rb is i_c, cb is k_c
                        i_c = rb
                        k_c = cb
                        i_glob = I * block_size + i_c
                        k_glob = J * block_size + k_c
                        W[k_glob * n + i_glob][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
