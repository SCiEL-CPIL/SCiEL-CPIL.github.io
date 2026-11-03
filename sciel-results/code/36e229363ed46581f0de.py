import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition using Strassen's algorithm
    # We construct the factor matrices U, V, W
    
    # Strassen 2x2 parameters
    # r=0: (a00+a11)(b00+b11)
    # r=1: (a10+a11)b00
    # r=2: a00(b01-b11)
    # r=3: a11(b10-b00)
    # r=4: (a00+a01)b11
    # r=5: (a10-a00)(b00+b01)
    # r=6: (a01-a11)(b10+b11)
    
    # Define the local Strassen factors for 2x2
    # U_loc[i][r], V_loc[i][r], W_loc[i][r]
    # i in 0..3 corresponds to a00, a01, a10, a11 (row-major)
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # r=0
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # r=1
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # r=2
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # r=3
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # r=4
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # r=5
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # r=6
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    total_rank = 56
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # We have 8 blocks of 2x2 Strassen
    # 2 blocks in A dimension (col halves)
    # 4 blocks in B dimension (row halves, col quarters)
    
    for half_a in range(2):
        for half_b_row in range(2):
            for jj in range(4): # 4 col blocks in B? No, B is 4x8.
                # B rows: 2*half_b_row, 2*half_b_row+1
                # B cols: 2*jj, 2*jj+1
                base_r = (half_a * 8 + half_b_row * 4 + jj) * 7
                
                # A indices
                # a00: row 0, col 2*half_a
                # a01: row 0, col 2*half_a+1
                # a10: row 1, col 2*half_a
                # a11: row 1, col 2*half_a+1
                a_indices = [
                    0 * m + (2 * half_a),
                    0 * m + (2 * half_a + 1),
                    1 * m + (2 * half_a),
                    1 * m + (2 * half_a + 1)
                ]
                
                # B indices
                # b00: row 2*half_b_row, col 2*jj
                # b01: row 2*half_b_row, col 2*jj+1
                # b10: row 2*half_b_row+1, col 2*jj
                # b11: row 2*half_b_row+1, col 2*jj+1
                b_indices = [
                    (2 * half_b_row) * p + (2 * jj),
                    (2 * half_b_row) * p + (2 * jj + 1),
                    (2 * half_b_row + 1) * p + (2 * jj),
                    (2 * half_b_row + 1) * p + (2 * jj + 1)
                ]
                
                # C indices
                # c00: row 2*jj, col 0
                # c01: row 2*jj+1, col 0
                # c10: row 2*jj, col 1
                # c11: row 2*jj+1, col 1
                # Note: C is 2x8. Row is col of C (0..7), Col is row of C (0..1)
                # W[k*n+i, r] -> k is col of C, i is row of C.
                # So index is col*n + row.
                c_indices = [
                    (2 * jj) * n + 0,
                    (2 * jj + 1) * n + 0,
                    (2 * jj) * n + 1,
                    (2 * jj + 1) * n + 1
                ]
                
                for r in range(7):
                    rr = base_r + r
                    for i in range(4):
                        val_u = U_loc[i, r]
                        if val_u != 0:
                            U[a_indices[i], rr] += val_u
                        val_v = V_loc[i, r]
                        if val_v != 0:
                            V[b_indices[i], rr] += val_v
                        val_w = W_loc[i, r]
                        if val_w != 0:
                            W[c_indices[i], rr] += val_w

    if time.time() - start_time > 110:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
