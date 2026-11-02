import numpy as np
import time

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    DEADLINE = start_time + 130

    # The decomposition is based on 6 Strassen 2x2x2 blocks.
    # Rank = 6 * 7 = 42.
    rank = 42
    
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        # A_idx: [a, b, c, d] corresponding to 2x2 matrix [[a,b],[c,d]]
        # B_idx: [e, f, g, h] corresponding to 2x2 matrix [[e,f],[g,h]]
        # C_idx: [c11, c12, c21, c22] corresponding to 2x2 matrix [[c11,c12],[c21,c22]]
        
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1 = (a+d)(e+h)
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    # Loop over 3 column pairs of B (and C)
    for k in range(3):
        # Block k of B is columns 2k, 2k+1.
        # B is 4x6. Indices: row * 6 + col.
        
        # A is 2x4.
        # Left part of A: cols 0, 1. Indices: 0, 1, 4, 5.
        # Right part of A: cols 2, 3. Indices: 2, 3, 6, 7.
        
        A_L_idx = [0, 1, 4, 5]
        A_R_idx = [2, 3, 6, 7]
        
        # Top part of B_k: rows 0, 1. Cols 2k, 2k+1.
        # Indices: 0*6+2k, 0*6+2k+1, 1*6+2k, 1*6+2k+1
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        
        # Bottom part of B_k: rows 2, 3. Cols 2k, 2k+1.
        # Indices: 2*6+2k, 2*6+2k+1, 3*6+2k, 3*6+2k+1
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        # C is 2x6.
        # Block k of C: cols 2k, 2k+1.
        # W index is k*n + i. Here n=2.
        # C[0, 2k] -> W[2k*2 + 0]
        # C[0, 2k+1] -> W[2k*2 + 1]
        # C[1, 2k] -> W[(2k+1)*2 + 0]
        # C[1, 2k+1] -> W[(2k+1)*2 + 1]
        C_idx = [2*k*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 0, (2*k+1)*2 + 1]
        
        # C_k = A_L * B_Tk + A_R * B_Bk
        
        # Term 1: A_L * B_Tk
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        # Term 2: A_R * B_Bk
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)

    # The rank is exactly 42.
    # We return the matrices.
    
    return {"U": U, "V": V, "W": W}
