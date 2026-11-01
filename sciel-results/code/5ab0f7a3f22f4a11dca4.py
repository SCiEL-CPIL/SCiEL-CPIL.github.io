"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Strassen-based decomposition for M_{2,2,16}
    # Rank = 8 blocks * 7 multiplications/block = 56
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2x2 block
    # U coeffs for a, b, c, d
    # V coeffs for e, f, g, h
    # W coeffs for c11, c12, c21, c22
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V coeffs for e, f, g, h
    # Note: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    strassen_V = [
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coeffs for c11, c12, c21, c22
    strassen_W = [
        [1, 0, 0, 1], # M1 -> c11, c22
        [0, 0, 1, -1],# M2 -> c21, c22
        [0, 1, 0, 1], # M3 -> c12, c22
        [1, 0, 1, 0], # M4 -> c11, c21
        [-1, 1, 0, 0],# M5 -> c11, c12
        [0, 0, 0, 1], # M6 -> c22
        [1, 0, 0, 0]  # M7 -> c11
    ]
    
    # Map local indices to global indices
    # U: a=0, b=1, c=2, d=3
    # V: e=2b, f=2b+1, g=16+2b, h=16+2b+1
    # W: c11=4b, c12=4b+2, c21=4b+1, c22=4b+3
    # Note: W index is k*n + i.
    # c11 is C[0, 2b] -> k=2b, i=0 -> 4b
    # c12 is C[0, 2b+1] -> k=2b+1, i=0 -> 4b+2
    # c21 is C[1, 2b] -> k=2b, i=1 -> 4b+1
    # c22 is C[1, 2b+1] -> k=2b+1, i=1 -> 4b+3
    
    start_time = time.time()
    deadline = 145.0 # 150s limit, 5s margin
    
    for b in range(8):
        if time.time() > deadline:
            break
            
        # Local indices for V and W
        v_e = 2 * b
        v_f = 2 * b + 1
        v_g = 16 + 2 * b
        v_h = 16 + 2 * b + 1
        
        w_c11 = 4 * b
        w_c12 = 4 * b + 2
        w_c21 = 4 * b + 1
        w_c22 = 4 * b + 3
        
        for k in range(7):
            col = b * 7 + k
            
            # Fill U
            for u_idx in range(4):
                U[u_idx][col] = strassen_U[k][u_idx]
                
            # Fill V
            V[v_e][col] = strassen_V[k][0]
            V[v_f][col] = strassen_V[k][1]
            V[v_g][col] = strassen_V[k][2]
            V[v_h][col] = strassen_V[k][3]
            
            # Fill W
            W[w_c11][col] = strassen_W[k][0]
            W[w_c12][col] = strassen_W[k][1]
            W[w_c21][col] = strassen_W[k][2]
            W[w_c22][col] = strassen_W[k][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
