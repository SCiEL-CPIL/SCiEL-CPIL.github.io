import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-44 matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 * 2x8)
    # A1 is A[0:2, 0:2]
    # B1 is B[0:2, 0:8]
    
    # Strassen coefficients for 2x2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U indices for A1 (2x2):
    # a: A[0,0] -> U[0]
    # b: A[0,1] -> U[1]
    # c: A[1,0] -> U[3]
    # d: A[1,1] -> U[4]
    
    # For each of the 4 blocks of B1 (2x2 each):
    for j in range(4):
        # V indices for B1 block j (2x2):
        # e: B[0, 2j] -> V[0 + 2j]
        # f: B[0, 2j+1] -> V[1 + 2j]
        # g: B[1, 2j] -> V[8 + 2j]
        # h: B[1, 2j+1] -> V[9 + 2j]
        
        # W indices for C block j (2x2):
        # R00: C[0, 2j] -> W[0 + 2j] (k=2j, i=0 -> 2j*2+0)
        # R01: C[0, 2j+1] -> W[1 + 2j] (k=2j+1, i=0 -> (2j+1)*2+0 = 4j+2? No. W index is k*n + i. n=2.
        # W[k*n + i].
        # C[i, k].
        # R00 is C[0, 2j]. k=2j, i=0. W[2j*2 + 0] = W[4j].
        # R01 is C[0, 2j+1]. k=2j+1, i=0. W[(2j+1)*2 + 0] = W[4j+2].
        # R10 is C[1, 2j]. k=2j, i=1. W[2j*2 + 1] = W[4j+1].
        # R11 is C[1, 2j+1]. k=2j+1, i=1. W[(2j+1)*2 + 1] = W[4j+3].
        
        w_r00 = 4*j
        w_r01 = 4*j + 2
        w_r10 = 4*j + 1
        w_r11 = 4*j + 3
        
        v_e = 2*j
        v_f = 2*j + 1
        v_g = 8 + 2*j
        v_h = 9 + 2*j
        
        # P1
        U[0][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P2
        U[3][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        W[w_r10][col] = 1
        W[w_r11][col] = -1
        col += 1
        
        # P3
        U[0][col] = 1
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_r01][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P4
        U[4][col] = 1
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_r00][col] = 1
        W[w_r10][col] = 1
        col += 1
        
        # P5
        U[0][col] = 1
        U[1][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = -1
        W[w_r01][col] = 1
        col += 1
        
        # P6
        U[3][col] = 1
        U[0][col] = -1
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P7
        U[1][col] = 1
        U[4][col] = -1
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        col += 1
        
    # Part 2: A2 * B2 (2x1 * 1x8)
    # A2 is A[0:2, 2]
    # B2 is B[2, 0:8]
    
    for i in range(2):
        for k in range(8):
            # U index for A[i, 2]
            u_idx = i * 3 + 2
            # V index for B[2, k]
            v_idx = 2 * 8 + k
            # W index for C[i, k]
            w_idx = k * 2 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    if col != rank:
        # Fallback to schoolbook if construction failed (should not happen)
        return {"U": [[0]*48 for _ in range(6)], "V": [[0]*48 for _ in range(24)], "W": [[0]*48 for _ in range(16)]}
        
    return {"U": U, "V": V, "W": W}
