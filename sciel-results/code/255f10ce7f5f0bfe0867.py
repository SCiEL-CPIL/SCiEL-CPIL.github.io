"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen (rank 7) for 2x2 blocks of size 3x3.
    # Each 3x3 block multiplication uses Laderman (rank 23).
    # Total rank = 7 * 23 = 161.
    
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # Blocks: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # U_S: (4, 7), V_S: (4, 7), W_S: (4, 7)
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # 00
        [0, 0, 0, 0, 1, 0, 1],  # 01
        [0, 1, -1, 0, 0, 1, 0], # 10
        [0, 1, 0, 1, 0, 0, -1]  # 11
    ]
    
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # 00
        [0, 0, 0, 1, 0, 1, 0],  # 01
        [0, 0, 1, 0, 0, 0, 1],  # 10
        [1, 0, 1, 0, 1, 0, 1]   # 11
    ]
    
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # 00
        [0, 0, 1, 0, 1, 0, 0],  # 01
        [0, 1, 0, 1, 0, 0, 0],  # 10
        [1, -1, 1, 0, 0, 1, 0]  # 11
    ]
    
    # Laderman coefficients for 3x3 blocks
    # U_L: (9, 23), V_L: (9, 23), W_L: (9, 23)
    # Indices 0..8 correspond to 00, 01, 02, 10, 11, 12, 20, 21, 22
    
    # Laderman's 23 products
    # I will use a known valid set.
    
    U_L = [
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 00
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 01
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 02
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 11
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 12
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 20
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 22
    ]
    
    # This is just the identity for the first 9. The other 14 are combinations.
    # I need the full 23.
    
    # Due to complexity and risk of error in hardcoding Laderman, 
    # I will use the schoolbook for 3x3 (rank 27) if I can't be sure.
    # Rank 7 * 27 = 189.
    # This is still better than 216.
    
    # Let's use Schoolbook for 3x3 to be safe.
    # Rank = 7 * 27 = 189.
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook 3x3
    # U_L, V_L, W_L are 9x27 identity-like.
    # For each product A_ij B_jk -> C_ik.
    # There are 27 such products.
    
    U_L = [[0]*27 for _ in range(9)]
    V_L = [[0]*27 for _ in range(9)]
    W_L = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_L[i*3+j][col] = 1
                V_L[j*3+k][col] = 1
                W_L[k*3+i][col] = 1
                col += 1
    
    # Compose Strassen and Schoolbook-3x3
    for s in range(7):
        for l in range(27):
            c = s * 27 + l
            # U
            for i in range(6):
                for j in range(6):
                    a = i // 3
                    b = j // 3
                    i_loc = i % 3
                    j_loc = j % 3
                    U[i*6+j][c] = U_S[a*2+b][s] * U_L[i_loc*3+j_loc][l]
            
            # V
            for j in range(6):
                for k in range(6):
                    b = j // 3
                    c_idx = k // 3
                    j_loc = j % 3
                    k_loc = k % 3
                    V[j*6+k][c] = V_S[b*2+c_idx][s] * V_L[j_loc*3+k_loc][l]
            
            # W
            for k in range(6):
                for i in range(6):
                    a = i // 3
                    c_idx = k // 3
                    k_loc = k % 3
                    i_loc = i % 3
                    W[k*6+i][c] = W_S[a*2+c_idx][s] * W_L[k_loc*3+i_loc][l]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
