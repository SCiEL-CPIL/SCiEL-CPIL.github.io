"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = 120.0  # 120 seconds limit, keep safety margin
    
    n, m, p = DIMENSIONS
    # Rank 57 decomposition
    rank = 57
    
    # Initialize matrices as lists of lists for ease of integer/half-integer storage
    # U: (n*m, rank) = (10, 57)
    # V: (m*p, rank) = (30, 57)
    # W: (p*n, rank) = (12, 57)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen decomposition for A_L (2x2) * B_L (2x6)
    # Split B_L into 3 blocks of 2x2
    # A_L indices:
    # a = A[0,0] -> U[0]
    # b = A[0,1] -> U[1]
    # c = A[1,0] -> U[5]
    # d = A[1,1] -> U[6]
    
    # Strassen terms definitions
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for A (indices in U)
    # 0: a, 1: b, 5: c, 6: d
    a_idx = 0
    b_idx = 1
    c_idx = 5
    d_idx = 6
    
    # For each block t (0, 1, 2)
    for t in range(3):
        # B indices for this block
        # e = B[0, 2t] -> V[0*6 + 2t]
        # f = B[0, 2t+1] -> V[0*6 + 2t+1]
        # g = B[1, 2t] -> V[1*6 + 2t]
        # h = B[1, 2t+1] -> V[1*6 + 2t+1]
        
        e_idx = 0 * 6 + 2 * t
        f_idx = 0 * 6 + 2 * t + 1
        g_idx = 1 * 6 + 2 * t
        h_idx = 1 * 6 + 2 * t + 1
        
        # C indices for this block
        # C[0, 2t], C[0, 2t+1]
        # C[1, 2t], C[1, 2t+1]
        
        c_00_idx = 2 * t * 2 + 0
        c_01_idx = 2 * t * 2 + 1
        c_10_idx = (2 * t + 1) * 2 + 0
        c_11_idx = (2 * t + 1) * 2 + 1
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c_00_idx][col] += 1
        W[c_11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c_10_idx][col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c_01_idx][col] += 1
        col += 1
        
        # M4 = d(g-f)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[f_idx][col] -= 1
        W[c_00_idx][col] += 1
        W[c_10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c_00_idx][col] -= 1
        W[c_01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c_11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c_00_idx][col] += 1
        col += 1
        
    # Part 2: Schoolbook for A_R (2x3) * B_R (3x6)
    # A_R cols: 2, 3, 4
    # B_R rows: 2, 3, 4
    # C cols: 0..5
    
    for i in range(n): # 0, 1
        for j in range(2, 5): # 2, 3, 4
            for k in range(p): # 0..5
                U[i * m + j][col] += 1
                V[j * p + k][col] += 1
                W[k * n + i][col] += 1
                col += 1
                
    # Check time
    if time.time() > deadline - 10:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
