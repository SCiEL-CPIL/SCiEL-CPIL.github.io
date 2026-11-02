import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 18 decomposition using Strassen for 2x2x2 blocks and standard for 2x2x1.
    # Block 1: cols 0,1 (2x2) -> Rank 7
    # Block 2: cols 2,3 (2x2) -> Rank 7
    # Block 3: col 4 (2x1) -> Rank 4
    # Total Rank: 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(b_start_col, w_start_row, u_offset, v_offset, w_offset):
        # b_start_col is the first column index of B (0 or 2)
        # w_start_row is the first row index of W for this block (0 or 4)
        # u_offset, v_offset, w_offset are the starting rank indices
        
        # Strassen factors for 2x2x2
        # M1 = (a+d)(e+h)
        U[0][u_offset] = 1; U[3][u_offset] = 1
        V[b_start_col][v_offset] = 1 # e=B[0, k]
        V[b_start_col + 2][v_offset] = 1 # h=B[1, k+1] ?? No, indices are j*p+k.
        # Let's map indices carefully.
        # B rows: 0,1 for j=0; 2,3 for j=1? No.
        # m=2, p=5.
        # Row index = j*p + k.
        # j=0: 0,1,2,3,4
        # j=1: 5,6,7,8,9
        
        # Block 1 (k=0,1):
        # e=B[0,0] -> V[0]
        # f=B[0,1] -> V[1]
        # g=B[1,0] -> V[5]
        # h=B[1,1] -> V[6]
        
        # Block 2 (k=2,3):
        # e=B[0,2] -> V[2]
        # f=B[0,3] -> V[3]
        # g=B[1,2] -> V[7]
        # h=B[1,3] -> V[8]
        
        # Let's generalize for b_start_col k0.
        # e = B[0, k0] -> V[k0]
        # f = B[0, k0+1] -> V[k0+1]
        # g = B[1, k0] -> V[5+k0]
        # h = B[1, k0+1] -> V[5+k0+1]
        
        # W rows:
        # C[0, k0] -> W[2*k0]
        # C[1, k0] -> W[2*k0+1]
        # C[0, k0+1] -> W[2*(k0+1)]
        # C[1, k0+1] -> W[2*(k0+1)+1]
        
        # M1 = (a+d)(e+h)
        U[0][u_offset] = 1; U[3][u_offset] = 1
        V[k0][v_offset] = 1; V[5+k0+1][v_offset] = 1 # h is B[1, k0+1] -> 5 + k0 + 1
        W[2*k0][w_offset] = 1 # C[0, k0] part 1
        W[2*(k0+1)+1][w_offset] = 1 # C[1, k0+1] part 1
        
        # M2 = (c+d)e
        U[2][u_offset+1] = 1; U[3][u_offset+1] = 1
        V[k0][v_offset+1] = 1
        W[2*k0+1][w_offset+1] = 1 # C[1, k0] part 1
        W[2*(k0+1)+1][w_offset+1] = -1 # C[1, k0+1] part -1
        
        # M3 = a(f-h)
        U[0][u_offset+2] = 1
        V[k0+1][v_offset+2] = 1 # f
        V[5+k0+1][v_offset+2] = -1 # -h
        W[2*(k0+1)][w_offset+2] = 1 # C[0, k0+1] part 1
        W[2*(k0+1)+1][w_offset+2] = 1 # C[1, k0+1] part 1
        
        # M4 = d(g-e)
        U[3][u_offset+3] = 1
        V[5+k0][v_offset+3] = 1 # g
        V[k0][v_offset+3] = -1 # -e
        W[2*k0][w_offset+3] = 1 # C[0, k0] part 1
        W[2*k0+1][w_offset+3] = 1 # C[1, k0] part 1
        
        # M5 = (a+b)h
        U[0][u_offset+4] = 1; U[1][u_offset+4] = 1
        V[5+k0+1][v_offset+4] = 1 # h
        W[2*k0][w_offset+4] = -1 # C[0, k0] part -1
        W[2*(k0+1)][w_offset+4] = 1 # C[0, k0+1] part 1
        
        # M6 = (c-a)(e+f)
        U[2][u_offset+5] = 1; U[0][u_offset+5] = -1
        V[k0][v_offset+5] = 1 # e
        V[k0+1][v_offset+5] = 1 # f
        W[2*(k0+1)+1][w_offset+5] = 1 # C[1, k0+1] part 1
        
        # M7 = (b-d)(g+h)
        U[1][u_offset+6] = 1; U[3][u_offset+6] = -1
        V[5+k0][v_offset+6] = 1 # g
        V[5+k0+1][v_offset+6] = 1 # h
        W[2*k0][w_offset+6] = 1 # C[0, k0] part 1
        
    # Block 1: k0=0, rank 0-6
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: k0=2, rank 7-13
    add_strassen_block(2, 4, 7, 7, 7)
    
    # Block 3: k0=4, standard 2x2x1, rank 14-17
    # C[0,4] = a B[0,4] + b B[1,4]
    # C[1,4] = c B[0,4] + d B[1,4]
    # B[0,4] -> V[4]
    # B[1,4] -> V[9]
    # C[0,4] -> W[8]
    # C[1,4] -> W[9]
    
    # M8 = a B[0,4]
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1
    
    # M9 = b B[1,4]
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1
    
    # M10 = c B[0,4]
    U[2][16] = 1
    V[4][16] = 1
    W[9][16] = 1
    
    # M11 = d B[1,4]
    U[3][17] = 1
    V[9][17] = 1
    W[9][17] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
