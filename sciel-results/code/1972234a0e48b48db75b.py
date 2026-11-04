import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100

    # Rank breakdown:
    # A11 (1x1) * B_top (1x8) -> Rank 8
    # A12 (1x2) * B_bot (2x8) -> Rank 16
    # A21 (2x1) * B_top (1x8) -> Rank 16
    # A22 (2x2) * B_bot (2x8) -> Rank 28 (4 blocks of Strassen 2x2x2)
    # Total Rank = 68
    
    total_rank = 68
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11 * B_top
    # A11 is A[0,0] (idx 0)
    # B_top is B[0,k] (idx k)
    # C[0,k] (idx 3k)
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[3 * k][col] = 1
        col += 1
        
    # 2. A12 * B_bot
    # A12 is A[0,1] (idx 1), A[0,2] (idx 2)
    # B_bot is B[1,k] (idx 8+k), B[2,k] (idx 16+k)
    # C[0,k] (idx 3k)
    for j in [1, 2]:
        for k in range(p):
            u_idx = j  # A[0, j]
            v_idx = (j - 1) * p + k  # B[j-1, k] (0-based row in B_bot is j-1)
            w_idx = 3 * k  # C[0, k]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # 3. A21 * B_top
    # A21 is A[1,0] (idx 3), A[2,0] (idx 6)
    # B_top is B[0,k] (idx k)
    # C[i,k] (idx 3k+i) for i in 1,2
    for i in [1, 2]:
        for k in range(p):
            u_idx = i * m  # A[i, 0]
            v_idx = k  # B[0, k]
            w_idx = 3 * k + i  # C[i, k]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # 4. A22 * B_bot
    # A22 is A[1,1] (idx 4), A[1,2] (idx 5), A[2,1] (idx 7), A[2,2] (idx 8)
    # B_bot is B[1,k] (idx 8+k), B[2,k] (idx 16+k)
    # C_bot is C[i,k] (idx 3k+i) for i in 1,2
    # Split into 4 blocks of 2x2
    for q in range(4):
        # Block q uses B columns 2q, 2q+1
        # Local indices for A22:
        # A11 -> 4, A12 -> 5, A21 -> 7, A22 -> 8
        a11 = 4
        a12 = 5
        a21 = 7
        a22 = 8
        
        # Local indices for B_bot block q:
        # B11 -> 8 + 2q, B12 -> 8 + 2q + 1
        # B21 -> 16 + 2q, B22 -> 16 + 2q + 1
        b11 = 8 + 2 * q
        b12 = 8 + 2 * q + 1
        b21 = 16 + 2 * q
        b22 = 16 + 2 * q + 1
        
        # Local indices for C_bot block q:
        # C11 -> 3(2q) + 1, C12 -> 3(2q+1) + 1
        # C21 -> 3(2q) + 2, C22 -> 3(2q+1) + 2
        c11 = 6 * q + 1
        c12 = 6 * q + 4
        c21 = 6 * q + 2
        c22 = 6 * q + 5
        
        # Strassen factors for 2x2x2
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A11+A22)B21 -> C12 += M2
        # M3 = A11(B11-B21) -> C11 += M3, C21 -= M3
        # M4 = A22(B22-B12) -> C22 += M4, C12 -= M4
        # M5 = (A21+A11)B12 -> C21 += M5
        # M6 = (A22-A11)(B11+B12) -> C22 += M6, C11 -= M6
        # M7 = (A11-A21)(B21+B22) -> C12 += M7, C21 -= M7
        
        # We add 7 terms.
        # Term 1: (A11+A22)(B11+B22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        col += 1
        
        # Term 2: (A11+A22)B21
        U[a11][col] = 1
        U[a22][col] = 1
        V[b21][col] = 1
        W[c12][col] = 1
        col += 1
        
        # Term 3: A11(B11-B21)
        U[a11][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        W[c11][col] = 1
        W[c21][col] = -1
        col += 1
        
        # Term 4: A22(B22-B12)
        U[a22][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[c22][col] = 1
        W[c12][col] = -1
        col += 1
        
        # Term 5: (A21+A11)B12
        U[a21][col] = 1
        U[a11][col] = 1
        V[b12][col] = 1
        W[c21][col] = 1
        col += 1
        
        # Term 6: (A22-A11)(B11+B12)
        U[a22][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        W[c11][col] = -1
        col += 1
        
        # Term 7: (A11-A21)(B21+B22)
        U[a11][col] = 1
        U[a21][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c12][col] = 1
        W[c21][col] = -1
        col += 1

    return {"U": U, "V": V, "W": W}
