import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Indices for 2x2: 0,1,2,3
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A11+A22, A21+A22, A11, A22, A11+A12, A21-A11, A12-A22
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B11+B22, B11, B12-B22, B21-B11, B22, B11+B12, B21+B22
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C (cols 0,1)
    # U indices: 0,1,2,3
    # V indices: 0,1,2,3
    # W indices: 0,1,2,3
    for r in range(7):
        for u_idx in range(4):
            U[u_idx][r] = U_s[u_idx][r]
        for v_idx in range(4):
            V[v_idx][r] = V_s[v_idx][r]
        for w_idx in range(4):
            W[w_idx][r] = W_s[w_idx][r]
            
    # Block 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3) -> C (cols 2,3)
    # U indices: 0,1,2,3
    # V indices: 2,3,6,7
    # W indices: 4,5,6,7
    for r in range(7, 14):
        for u_idx in range(4):
            U[u_idx][r] = U_s[u_idx][r-7]
        v_map = [2, 3, 6, 7]
        for v_idx in range(4):
            V[v_map[v_idx]][r] = V_s[v_idx][r-7]
        w_map = [4, 5, 6, 7]
        for w_idx in range(4):
            W[w_map[w_idx]][r] = W_s[w_idx][r-7]
            
    # Block 3: A2 (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C (cols 0,1)
    # U indices: 2,3,6,7
    # V indices: 8,9,12,13
    # W indices: 0,1,2,3
    for r in range(14, 21):
        u_map = [2, 3, 6, 7]
        for u_idx in range(4):
            U[u_map[u_idx]][r] = U_s[u_idx][r-14]
        v_map = [8, 9, 12, 13]
        for v_idx in range(4):
            V[v_map[v_idx]][r] = V_s[v_idx][r-14]
        for w_idx in range(4):
            W[w_idx][r] = W_s[w_idx][r-14]
            
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3, cols 2,3) -> C (cols 2,3)
    # U indices: 2,3,6,7
    # V indices: 10,11,14,15
    # W indices: 4,5,6,7
    for r in range(21, 28):
        u_map = [2, 3, 6, 7]
        for u_idx in range(4):
            U[u_map[u_idx]][r] = U_s[u_idx][r-21]
        v_map = [10, 11, 14, 15]
        for v_idx in range(4):
            V[v_map[v_idx]][r] = V_s[v_idx][r-21]
        w_map = [4, 5, 6, 7]
        for w_idx in range(4):
            W[w_map[w_idx]][r] = W_s[w_idx][r-21]
            
    return {"U": U, "V": V, "W": W}
