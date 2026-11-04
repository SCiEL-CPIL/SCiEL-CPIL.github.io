"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 16 Strassen blocks (2x2x2) for 4x4x8 part: 16 * 7 = 112
    # 16 Schoolbook terms for 4x4x1 part: 16
    # Total Rank = 128
    rank = 128
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 128
    V = [[0] * rank for _ in range(m * p)] # 36 x 128
    W = [[0] * rank for _ in range(p * n)] # 36 x 128
    
    # Strassen coefficients for 2x2x2
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # U_s[k, r] where k is rank index 0-6, r is A index 0-3
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_s = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    
    # V_s[k, r] where k is rank index 0-6, r is B index 0-3
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_s = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, -1, 1, 0], # m3
        [-1, 1, 0, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    
    # W_s[k, r] where k is rank index 0-6, r is C index 0-3
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    W_s = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1], # m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0], # m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]

    col_offset = 0
    
    # Part 1: 4x4x8 using 16 Strassen blocks
    # A is 4x4. Split into 2x2 blocks A_r_s for r,s in {0,1}
    # B is 4x8. Split into 2x2 blocks B_s_c for s in {0,1}, c in {0,1,2,3}
    # C is 4x8. Split into 2x2 blocks C_r_c for r in {0,1}, c in {0,1,2,3}
    
    for r in range(2):
        for s in range(2):
            for c in range(4):
                # This term computes A_r_s * B_s_c -> C_r_c
                # A block rows: 2r, 2r+1. Cols: 2s, 2s+1.
                # B block rows: 2s, 2s+1. Cols: 2c, 2c+1.
                # C block rows: 2r, 2r+1. Cols: 2c, 2c+1.
                
                for k in range(7):
                    # Add to U
                    for ai in range(2):
                        for aj in range(2):
                            val = U_s[k][ai*2+aj]
                            if val != 0:
                                global_u_idx = (2*r + ai) * 4 + (2*s + aj)
                                U[global_u_idx][col_offset + k] += val
                            
                    # Add to V
                    for bi in range(2):
                        for bj in range(2):
                            val = V_s[k][bi*2+bj]
                            if val != 0:
                                global_v_idx = (2*s + bi) * 9 + (2*c + bj)
                                V[global_v_idx][col_offset + k] += val
                                
                    # Add to W
                    for ci in range(2):
                        for cj in range(2):
                            val = W_s[k][ci*2+cj]
                            if val != 0:
                                global_w_idx = (2*r + ci) * 4 + (2*c + cj)
                                W[global_w_idx][col_offset + k] += val
                                
                col_offset += 7

    # Part 2: 4x4x1 using Schoolbook
    # C_i_8 = sum_j A_i_j B_j_8
    for i in range(4):
        for j in range(4):
            # U
            U[i * 4 + j][col_offset] = 1
            # V
            V[j * 9 + 8][col_offset] = 1
            # W
            W[8 * 4 + i][col_offset] = 1
            col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
