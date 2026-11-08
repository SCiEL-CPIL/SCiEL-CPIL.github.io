import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_indices, v_indices, w_indices, start_col):
        # u_indices: [x00, x01, x10, x11] global row indices in U
        # v_indices: [y00, y01, y10, y11] global row indices in V
        # w_indices: [z00, z01, z10, z11] global row indices in W
        # Strassen terms:
        # M1 = (x00 + x11)(y00 + y11) -> z00 += 1, z11 += 1
        # M2 = (x10 + x11)y00       -> z10 += 1, z11 -= 1
        # M3 = x00(y10 - y11)       -> z01 += 1, z11 += 1
        # M4 = x11(y01 - y00)       -> z00 += 1, z10 += 1
        # M5 = (x00 + x01)y11       -> z00 -= 1, z01 += 1
        # M6 = (x10 - x00)(y00 + y01) -> z11 += 1
        # M7 = (x01 - x11)(y10 + y11) -> z00 += 1
        
        x00, x01, x10, x11 = u_indices
        y00, y01, y10, y11 = v_indices
        z00, z01, z10, z11 = w_indices
        
        # M1
        col = start_col
        U[x00][col] += 1
        U[x11][col] += 1
        V[y00][col] += 1
        V[y11][col] += 1
        W[z00][col] += 1
        W[z11][col] += 1
        
        # M2
        col = start_col + 1
        U[x10][col] += 1
        U[x11][col] += 1
        V[y00][col] += 1
        W[z10][col] += 1
        W[z11][col] -= 1
        
        # M3
        col = start_col + 2
        U[x00][col] += 1
        V[y10][col] += 1
        V[y11][col] -= 1
        W[z01][col] += 1
        W[z11][col] += 1
        
        # M4
        col = start_col + 3
        U[x11][col] += 1
        V[y01][col] += 1
        V[y00][col] -= 1
        W[z00][col] += 1
        W[z10][col] += 1
        
        # M5
        col = start_col + 4
        U[x00][col] += 1
        U[x01][col] += 1
        V[y11][col] += 1
        W[z00][col] -= 1
        W[z01][col] += 1
        
        # M6
        col = start_col + 5
        U[x10][col] += 1
        U[x00][col] -= 1
        V[y00][col] += 1
        V[y01][col] += 1
        W[z11][col] += 1
        
        # M7
        col = start_col + 6
        U[x01][col] += 1
        U[x11][col] -= 1
        V[y10][col] += 1
        V[y11][col] += 1
        W[z00][col] += 1

    # Block 1: A_L (i 0-1, j 0-1) * B_TL (j 0-1, k 0-1) -> C (i 0-1, k 0-1)
    # U rows: i*4+j. 0,1,4,5
    # V rows: j*5+k. 0,1,5,6
    # W rows: k*2+i. 0,2,1,3
    add_strassen_block(
        [0, 1, 4, 5],
        [0, 1, 5, 6],
        [0, 2, 1, 3],
        0
    )
    
    # Block 2: A_R (i 0-1, j 2-3) * B_BL (j 2-3, k 0-1) -> C (i 0-1, k 0-1)
    # U rows: 2,3,6,7
    # V rows: 10,11,15,16
    # W rows: 0,2,1,3
    add_strassen_block(
        [2, 3, 6, 7],
        [10, 11, 15, 16],
        [0, 2, 1, 3],
        7
    )
    
    # Block 3: A_L (i 0-1, j 0-1) * B_TR (j 0-1, k 2-3) -> C (i 0-1, k 2-3)
    # U rows: 0,1,4,5
    # V rows: 2,3,7,8
    # W rows: 4,6,5,7
    add_strassen_block(
        [0, 1, 4, 5],
        [2, 3, 7, 8],
        [4, 6, 5, 7],
        14
    )
    
    # Block 4: A_R (i 0-1, j 2-3) * B_BR (j 2-3, k 2-3) -> C (i 0-1, k 2-3)
    # U rows: 2,3,6,7
    # V rows: 12,13,17,18
    # W rows: 4,6,5,7
    add_strassen_block(
        [2, 3, 6, 7],
        [12, 13, 17, 18],
        [4, 6, 5, 7],
        21
    )
    
    # Block 5: A (i 0-1, j 0-3) * B_extra (j 0-3, k 4) -> C (i 0-1, k 4)
    col = 28
    for i in range(2):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
