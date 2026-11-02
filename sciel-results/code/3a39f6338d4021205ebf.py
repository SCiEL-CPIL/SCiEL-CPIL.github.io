import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Rank 56 decomposition based on 8 independent 2x2x2 Strassen blocks.
    # Total rank = 8 * 7 = 56.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)] # 16 x 56
    
    # Strassen components for 2x2 matrix multiplication
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # Indices: 0..3 for flat 2x2 matrices
    # A: i*2+j, B: j*2+k, C: i*2+k (local)
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1

    # Mapping logic
    # j: 0..3 (col block of C, corresponds to 2 columns of C, 2 cols of B)
    # s: 0..1 (selector for A col-block / B row-block)
    # s=0: A cols 0,1; B rows 0,1
    # s=1: A cols 2,3; B rows 2,3
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # A mapping: local 0..3 -> global
            # Local i (0,1), j_loc (0,1) -> i*2 + j_loc
            # Global i (0,1), j_g (0..3) -> i*4 + j_g
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0
                a_map[1] = 0*4 + 1
                a_map[2] = 1*4 + 0
                a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            # B mapping: local 0..3 -> global
            # Local j_loc (0,1), k_loc (0,1) -> j_loc*2 + k_loc
            # Global j_g (0..3), k_g (0..7) -> j_g*8 + k_g
            b_map = [0]*4
            if s == 0:
                # Rows 0,1
                b_map[0] = 0*8 + (2*j)
                b_map[1] = 0*8 + (2*j+1)
                b_map[2] = 1*8 + (2*j)
                b_map[3] = 1*8 + (2*j+1)
            else:
                # Rows 2,3
                b_map[0] = 2*8 + (2*j)
                b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j)
                b_map[3] = 3*8 + (2*j+1)
                
            # W mapping: local 0..3 -> global
            # Local i (0,1), k_loc (0,1) -> i*2 + k_loc
            # Global i (0,1), k_g (0..7)
            # W Index Convention: k_g * n + i_g = k_g * 2 + i_g
            c_map = [0]*4
            # k_g is 2j or 2j+1
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1

            # Fill U, V, W
            for l in range(7):
                r = base_r + l
                
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        ga = a_map[la]
                        U[ga][r] += val
                        
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        gb = b_map[lb]
                        V[gb][r] += val
                        
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        gc = c_map[lc]
                        W[gc][r] += val
                        
    if time.time() - start_time > 140:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
