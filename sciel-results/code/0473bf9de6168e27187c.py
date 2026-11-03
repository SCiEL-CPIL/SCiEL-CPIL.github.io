import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 combinations
    # Each entry is a list of (local_row, local_col, coeff)
    # Local indices 0, 1
    # M1 = (A00 + A11)(B00 + B11)
    A_s0 = [(0,0,1), (1,1,1)]
    B_s0 = [(0,0,1), (1,1,1)]
    C_s0 = [(0,0,1), (1,1,1)] # z00 += M1, z11 += M1
    
    # M2 = (A10 + A11)B00
    A_s1 = [(1,0,1), (1,1,1)]
    B_s1 = [(0,0,1)]
    C_s1 = [(1,0,1), (1,1,-1)] # z10 += M2, z11 -= M2
    
    # M3 = A00(B01 - B11)
    A_s2 = [(0,0,1)]
    B_s2 = [(0,1,1), (1,1,-1)]
    C_s2 = [(0,1,1), (1,1,1)] # z01 += M3, z11 += M3
    
    # M4 = A11(B10 - B00)
    A_s3 = [(1,1,1)]
    B_s3 = [(1,0,1), (0,0,-1)]
    C_s3 = [(0,0,1), (1,0,1)] # z00 += M4, z10 += M4
    
    # M5 = (A00 + A01)B11
    A_s4 = [(0,0,1), (0,1,1)]
    B_s4 = [(1,1,1)]
    C_s4 = [(0,0,-1), (0,1,1)] # z00 -= M5, z01 += M5
    
    # M6 = (A10 - A00)(B00 + B01)
    A_s5 = [(1,0,1), (0,0,-1)]
    B_s5 = [(0,0,1), (0,1,1)]
    C_s5 = [(1,1,1)] # z11 += M6
    
    # M7 = (A01 - A11)(B10 + B11)
    A_s6 = [(0,1,1), (1,1,-1)]
    B_s6 = [(1,0,1), (1,1,1)]
    C_s6 = [(0,0,1)] # z00 += M7
    
    A_combs = [A_s0, A_s1, A_s2, A_s3, A_s4, A_s5, A_s6]
    B_combs = [B_s0, B_s1, B_s2, B_s3, B_s4, B_s5, B_s6]
    C_combs = [C_s0, C_s1, C_s2, C_s3, C_s4, C_s5, C_s6]
    
    block_size = 2
    num_blocks = 3
    
    for i_b in range(num_blocks):
        for j_b in range(num_blocks):
            for k_b in range(num_blocks):
                for s in range(7):
                    col = ((i_b * num_blocks + j_b) * num_blocks + k_b) * 7 + s
                    
                    # Update U
                    for rb, cb, coeff in A_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = k_b * block_size + cb
                        U[i_glob * m + j_glob][col] += coeff
                        
                    # Update V
                    for rb, cb, coeff in B_combs[s]:
                        j_glob = k_b * block_size + rb
                        k_glob = j_b * block_size + cb
                        V[j_glob * p + k_glob][col] += coeff
                        
                    # Update W
                    for rb, cb, coeff in C_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = j_b * block_size + cb
                        W[j_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
