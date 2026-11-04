import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Rank 42 solution based on Strassen decomposition of 2x2 blocks
    # M_{2,4,6} is decomposed into 3 blocks of M_{2,4,2}
    # Each M_{2,4,2} is decomposed into 2 blocks of M_{2,2,2}
    # Each M_{2,2,2} uses Strassen's 7-term algorithm
    # Total rank = 3 * 2 * 7 = 42
    
    U = np.zeros((dim_A, 42), dtype=np.int32)
    V = np.zeros((dim_B, 42), dtype=np.int32)
    W = np.zeros((dim_C, 42), dtype=np.int32)
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        # a_idx: 4 indices into U (rows)
        # b_idx: 4 indices into V (rows)
        # w_idx: 4 indices into W (rows)
        # start_col: column offset in the rank dimension
        
        # Strassen's 7 multiplications for 2x2 matrices
        # Let A = [[a0, a1], [a2, a3]] and B = [[b0, b1], [b2, b3]]
        # M1 = (a0+a3)(b0+b3)
        # M2 = (a2+a3)b0
        # M3 = a0(b1-b3)
        # M4 = a3(b2-b0)
        # M5 = (a0+a1)b3
        # M6 = (a2-a0)(b0+b1)
        # M7 = (a1-a3)(b2+b3)
        
        # U entries (A matrix part)
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        U[a_idx[0]][start_col + 2] += 1
        U[a_idx[3]][start_col + 3] += 1
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        U[a_idx[2]][start_col + 5] += 1
        U[a_idx[0]][start_col + 5] -= 1
        U[a_idx[1]][start_col + 6] += 1
        U[a_idx[3]][start_col + 6] -= 1
        
        # V entries (B matrix part)
        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 1] += 1
        V[b_idx[1]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        V[b_idx[2]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        V[b_idx[3]][start_col + 4] += 1
        V[b_idx[0]][start_col + 5] += 1
        V[b_idx[1]][start_col + 5] += 1
        V[b_idx[2]][start_col + 6] += 1
        V[b_idx[3]][start_col + 6] += 1
        
        # W entries (C matrix part)
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[0]][start_col + 6] += 1
        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[1]][start_col + 4] += 1
        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[2]][start_col + 3] += 1
        W[w_idx[3]][start_col + 0] += 1
        W[w_idx[3]][start_col + 1] -= 1
        W[w_idx[3]][start_col + 2] += 1
        W[w_idx[3]][start_col + 5] += 1

    # Define index mappings
    # A is 2x4. Split into two 2x2 blocks: Left (cols 0,1) and Right (cols 2,3)
    # A_L indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # A_R indices: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7
    a_L = [0, 1, 2, 3]
    a_R = [4, 5, 6, 7]
    
    # B is 4x6. Split rows into Top (0,1) and Bottom (2,3). Split cols into 3 blocks of 2.
    # Block k (0,1,2) has cols 2k, 2k+1.
    # For each block k, we have B_Tk (2x2) and B_Bk (2x2).
    # B_Tk indices:
    #   k=0: (0,0)->0, (0,1)->1, (1,0)->6, (1,1)->7  (Note: B row j, col k. Index j*6+k)
    #   Wait, B is 4x6. Index b = j*6 + k.
    #   B_Tk (rows 0,1; cols 2k, 2k+1):
    #   (0, 2k), (0, 2k+1), (1, 2k), (1, 2k+1)
    #   Indices: 0*6+2k, 0*6+2k+1, 1*6+2k, 1*6+2k+1
    #   B_Bk (rows 2,3; cols 2k, 2k+1):
    #   (2, 2k), (2, 2k+1), (3, 2k), (3, 2k+1)
    #   Indices: 2*6+2k, 2*6+2k+1, 3*6+2k, 3*6+2k+1
    
    # W is 6x2 (p*n). Index w = k*n + i = 2k + i.
    # For block k, output C_k is 2x2.
    # Indices: (2k, 0), (2k, 1), (2k+1, 0), (2k+1, 1)
    # w_idx for block k: [2k, 2k+1, 2k+2, 2k+3]
    
    # We have 3 blocks k=0,1,2.
    # For each k, we have two Strassen blocks: one for A_L * B_Tk and one for A_R * B_Bk.
    # Each Strassen block uses 7 ranks.
    # Total 3 * 2 * 7 = 42 ranks.
    
    for k in range(3):
        # B_Tk indices
        b_Tk = [2*k, 2*k+1, 6+2*k, 6+2*k+1]
        # B_Bk indices
        b_Bk = [12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1]
        
        # W_k indices
        w_k = [2*k, 2*k+1, 2*k+2, 2*k+3]
        
        # Strassen block 1: A_L * B_Tk
        # Offset: k * 14
        add_strassen_block(a_L, b_Tk, w_k, k * 14)
        
        # Strassen block 2: A_R * B_Bk
        # Offset: k * 14 + 7
        add_strassen_block(a_R, b_Bk, w_k, k * 14 + 7)
        
    return {"U": U, "V": V, "W": W}
