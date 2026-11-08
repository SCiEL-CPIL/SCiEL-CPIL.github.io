import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        for ui in u_idx:
            U[ui][col] += val
        for vi in v_idx:
            V[vi][col] += val
        for wi in w_idx:
            W[wi][col] += val
            # Wait, U, V, W are separate matrices.
            # U[ui, col] += val
            # V[vi, col] += val
            # W[wi, col] += val
            pass
        
    # Let's just fill them directly.
    
    # Part 1: Strassen for A1 (cols 0,1) * B1a (rows 0,1, cols 0,1)
    # A indices: i in {0,1}, j in {0,1} -> u = 3i + j
    # B indices: j in {0,1}, k in {0,1} -> v = 5j + k
    # C indices: k in {0,1}, i in {0,1} -> w = 2k + i
    
    # M1: (A00+A11)(B00+B11)
    col = 0
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    
    # M2: (A10+A11)B00
    col = 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    
    # M3: A00(B01-B11)
    col = 2
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    
    # M4: A11(B10-B00)
    col = 3
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[0][col] = 1
    
    # M5: (A00+A01)B11
    col = 4
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[2][col] = 1; W[0][col] = -1
    
    # M6: (A10-A00)(B00+B01)
    col = 5
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    
    # M7: (A01-A11)(B10+B11)
    col = 6
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    
    # Part 2: Schoolbook for A1 (cols 0,1) * B1b (rows 0,1, cols 2,3,4)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {2,3,4}
    # C indices: k in {2,3,4}, i in {0,1}
    
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u_idx = 3 * i + j
                v_idx = 5 * j + k
                w_idx = 2 * k + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Part 3: Schoolbook for A2 (col 2) * B2 (row 2, cols 0-4)
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..4}
    # C indices: k in {0..4}, i in {0,1}
    
    for i in range(2):
        for k in range(5):
            u_idx = 3 * i + 2
            v_idx = 5 * 2 + k
            w_idx = 2 * k + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
