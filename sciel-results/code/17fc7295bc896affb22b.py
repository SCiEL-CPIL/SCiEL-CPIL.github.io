import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1
        
    # Process A1 B1 (j in 0,1)
    # A1 is rows 0,1,2 cols 0,1
    # B1 is rows 0,1 cols 0..5
    
    # Part 1: a (row 0 of A1) * B1 blocks
    # a = A[0,0], A[0,1]
    # B1t is B[0..1, 2t..2t+1]
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        # B1t indices in V:
        # B[0, k0] -> 0*p + k0
        # B[0, k1] -> 0*p + k1
        # B[1, k0] -> 1*p + k0
        # B[1, k1] -> 1*p + k1
        
        # Term 1: A[0,0] * B[0, k0] * C[0, k0]
        add_term([0*m + 0], [1], [0*p + k0], [1], [k0*n + 0], [1])
        # Term 2: A[0,1] * B[1, k0] * C[0, k0]
        add_term([0*m + 1], [1], [1*p + k0], [1], [k0*n + 0], [1])
        # Term 3: A[0,0] * B[0, k1] * C[0, k1]
        add_term([0*m + 0], [1], [0*p + k1], [1], [k1*n + 0], [1])
        # Term 4: A[0,1] * B[1, k1] * C[0, k1]
        add_term([0*m + 1], [1], [1*p + k1], [1], [k1*n + 0], [1])
        
    # Part 2: A1' (rows 1,2 of A1) * B1 blocks
    # A1' = [[A[1,0], A[1,1]], [A[2,0], A[2,1]]]
    # B1t = [[B[0, k0], B[0, k1]], [B[1, k0], B[1, k1]]]
    # C1t = [[C[1, k0], C[1, k1]], [C[2, k0], C[2, k1]]]
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # A indices
        a11_idx = 1*m + 0
        a12_idx = 1*m + 1
        a21_idx = 2*m + 0
        a22_idx = 2*m + 1
        
        # B indices
        b11_idx = 0*p + k0
        b12_idx = 0*p + k1
        b21_idx = 1*p + k0
        b22_idx = 1*p + k1
        
        # C indices
        c11_idx = k0*n + 1
        c12_idx = k1*n + 1
        c21_idx = k0*n + 2
        c22_idx = k1*n + 2
        
        # M1 = (A11+A22)(B11+B22)
        add_term([a11_idx, a22_idx], [1, 1], [b11_idx, b22_idx], [1, 1], 
                 [c11_idx, c22_idx], [1, 1])
        
        # M2 = (A21+A22)B11
        add_term([a21_idx, a22_idx], [1, 1], [b11_idx], [1], 
                 [c21_idx], [1])
        
        # M3 = A11(B21-B22)
        add_term([a11_idx], [1], [b21_idx, b22_idx], [1, -1], 
                 [c12_idx, c22_idx], [1, 1])
        
        # M4 = A22(B11-B12)
        add_term([a22_idx], [1], [b11_idx, b12_idx], [1, -1], 
                 [c11_idx, c22_idx], [1, 1])
        
        # M5 = (A11+A12)B22
        add_term([a11_idx, a12_idx], [1, 1], [b22_idx], [1], 
                 [c11_idx, c22_idx], [-1, 1])
        
        # M6 = (A21-A11)(B11+B12)
        add_term([a21_idx, a11_idx], [1, -1], [b11_idx, b12_idx], [1, 1], 
                 [c21_idx], [1])
        
        # M7 = (A12-A22)(B21+B22)
        add_term([a12_idx, a22_idx], [1, -1], [b21_idx, b22_idx], [1, 1], 
                 [c12_idx, c21_idx, c22_idx], [1, -1, -1])

    # Process A2 B2 (j in 2,3)
    # A2 is rows 0,1,2 cols 2,3
    # B2 is rows 2,3 cols 0..5
    
    # Part 1: a (row 0 of A2) * B2 blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: A[0,2] * B[2, k0] * C[0, k0]
        add_term([0*m + 2], [1], [2*p + k0], [1], [k0*n + 0], [1])
        # Term 2: A[0,3] * B[3, k0] * C[0, k0]
        add_term([0*m + 3], [1], [3*p + k0], [1], [k0*n + 0], [1])
        # Term 3: A[0,2] * B[2, k1] * C[0, k1]
        add_term([0*m + 2], [1], [2*p + k1], [1], [k1*n + 0], [1])
        # Term 4: A[0,3] * B[3, k1] * C[0, k1]
        add_term([0*m + 3], [1], [3*p + k1], [1], [k1*n + 0], [1])
        
    # Part 2: A2' (rows 1,2 of A2) * B2 blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # A indices
        a11_idx = 1*m + 2
        a12_idx = 1*m + 3
        a21_idx = 2*m + 2
        a22_idx = 2*m + 3
        
        # B indices
        b11_idx = 2*p + k0
        b12_idx = 2*p + k1
        b21_idx = 3*p + k0
        b22_idx = 3*p + k1
        
        # C indices
        c11_idx = k0*n + 1
        c12_idx = k1*n + 1
        c21_idx = k0*n + 2
        c22_idx = k1*n + 2
        
        # M1 = (A11+A22)(B11+B22)
        add_term([a11_idx, a22_idx], [1, 1], [b11_idx, b22_idx], [1, 1], 
                 [c11_idx, c22_idx], [1, 1])
        
        # M2 = (A21+A22)B11
        add_term([a21_idx, a22_idx], [1, 1], [b11_idx], [1], 
                 [c21_idx], [1])
        
        # M3 = A11(B21-B22)
        add_term([a11_idx], [1], [b21_idx, b22_idx], [1, -1], 
                 [c12_idx, c22_idx], [1, 1])
        
        # M4 = A22(B11-B12)
        add_term([a22_idx], [1], [b11_idx, b12_idx], [1, -1], 
                 [c11_idx, c22_idx], [1, 1])
        
        # M5 = (A11+A12)B22
        add_term([a11_idx, a12_idx], [1, 1], [b22_idx], [1], 
                 [c11_idx, c22_idx], [-1, 1])
        
        # M6 = (A21-A11)(B11+B12)
        add_term([a21_idx, a11_idx], [1, -1], [b11_idx, b12_idx], [1, 1], 
                 [c21_idx], [1])
        
        # M7 = (A12-A22)(B21+B22)
        add_term([a12_idx, a22_idx], [1, -1], [b21_idx, b22_idx], [1, 1], 
                 [c12_idx, c21_idx, c22_idx], [1, -1, -1])

    return {"U": U, "V": V, "W": W}
