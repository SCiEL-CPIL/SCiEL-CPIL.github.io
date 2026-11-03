import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition using 8 blocks of Strassen (2x2x2)
    # Split A (2x4) into 2 blocks of 2x2: A0, A1
    # Split B (4x8) into 4 blocks of 2x2 per row-pair, total 8 blocks?
    # A = [A_left, A_right]
    # B = [B_top_left, B_top_right; B_bot_left, B_bot_right] ? No.
    # B is 4x8.
    # Let's split B into 4 blocks of 2x2 horizontally for the top 2 rows, and 4 for bottom 2 rows.
    # B = [B_{00} B_{01} B_{02} B_{03}; B_{10} B_{11} B_{12} B_{13}] for top 2 rows?
    # No, B is 4 rows.
    # B = [B_{00} B_{01} B_{02} B_{03}] where each is 1x2? No.
    
    # The previous logic:
    # A_left (cols 0,1), A_right (cols 2,3).
    # B_top (rows 0,1), B_bot (rows 2,3).
    # B_top is 2x8. Split into 4 blocks of 2x2: B_{t0}, B_{t1}, B_{t2}, B_{t3}.
    # B_bot is 2x8. Split into 4 blocks of 2x2: B_{b0}, B_{b1}, B_{b2}, B_{b3}.
    # C = A_left B_top + A_right B_bot.
    # C_col_j = A_left B_{tj} + A_right B_{bj}.
    # For each j, we compute A_left B_{tj} (rank 7) and A_right B_{bj} (rank 7).
    # Total 8 * 7 = 56.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    for s in range(4):
        for t in range(2):
            base_r = (s * 2 + t) * 7
            
            a_col0 = 2 * t
            a_col1 = 2 * t + 1
            b_row0 = 2 * t
            b_row1 = 2 * t + 1
            c_col0 = 2 * s
            c_col1 = 2 * s + 1
            
            u_x00 = 0 * 4 + a_col0
            u_x01 = 0 * 4 + a_col1
            u_x10 = 1 * 4 + a_col0
            u_x11 = 1 * 4 + a_col1
            
            v_y00 = b_row0 * 8 + c_col0
            v_y01 = b_row0 * 8 + c_col1
            v_y10 = b_row1 * 8 + c_col0
            v_y11 = b_row1 * 8 + c_col1
            
            w_z00 = c_col0 * 2 + 0
            w_z01 = c_col1 * 2 + 0
            w_z10 = c_col0 * 2 + 1
            w_z11 = c_col1 * 2 + 1
            
            # M1
            r = base_r
            U[u_x00][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += 1
            W[w_z11][r] += 1
            
            # M2
            r = base_r + 1
            U[u_x10][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            W[w_z10][r] += 1
            W[w_z11][r] += -1
            
            # M3
            r = base_r + 2
            U[u_x00][r] += 1
            V[v_y01][r] += 1
            V[v_y11][r] += -1
            W[w_z01][r] += 1
            W[w_z11][r] += 1
            
            # M4
            r = base_r + 3
            U[u_x11][r] += 1
            V[v_y10][r] += 1
            V[v_y00][r] += -1
            W[w_z00][r] += 1
            W[w_z10][r] += 1
            
            # M5
            r = base_r + 4
            U[u_x00][r] += 1
            U[u_x01][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += -1
            W[w_z01][r] += 1
            
            # M6
            r = base_r + 5
            U[u_x10][r] += 1
            U[u_x00][r] += -1
            V[v_y00][r] += 1
            V[v_y01][r] += 1
            W[w_z11][r] += 1
            
            # M7
            r = base_r + 6
            U[u_x01][r] += 1
            U[u_x11][r] += -1
            V[v_y10][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
