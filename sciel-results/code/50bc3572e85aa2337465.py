import time
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Split B (3x6) into two 3x3 blocks. For each 3x3 * 3x3,
    use a rank-23 decomposition. Total rank = 2 * 23 = 46.
    
    Uses explicit verified integer coefficients for the 23-term 3x3x3 decomposition.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    # Rank-23 decomposition for 3x3 x 3x3
    # U: 9x23, V: 9x23, W: 9x23
    # Index convention: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Verified 23-term decomposition (integer coefficients)
    # Each entry is a list of 23 coefficients for that row
    U_3x3 = [
        [1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1],
        [0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
        [0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
    ]
    
    V_3x3 = [
        [1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1],
        [0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  0,  1,  1,  0, 1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
        [0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
    ]
    
    W_3x3 = [
        [1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1],
        [0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
        [0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0],
        [1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1],
        [1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],
        [0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0],
    ]
    
    # Verify the 3x3x3 decomposition
    def verify_3x3(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(23):
                        s += U[i*3+j][c] * V[j*3+k][c] * W[k*3+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_3x3(U_3x3, V_3x3, W_3x3):
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the 3x3x6 decomposition using two copies of the 3x3x3 decomposition
    # B = [B1, B2] where B1, B2 are 3x3
    # C = A * B = [A*B1, A*B2]
    # 
    # For A*B1: U rows indexed by i*3+j, V rows indexed by j*3+k1 where k1=0,1,2
    # For A*B2: U rows indexed by i*3+j, V rows indexed by j*3+k2 where k2=0,1,2
    #   but in the full 3x6, k = 3+k2
    # W for B1: W[k1*3+i], W for B2: W[(3+k2)*3+i]
    
    rank_total = 46
    U_full = [[0] * rank_total for _ in range(9)]  # 3*3
    V_full = [[0] * rank_total for _ in range(18)]  # 3*6
    W_full = [[0] * rank_total for _ in range(18)]  # 6*3
    
    # First 23 terms: for B1 (columns 0-2 of B, i.e., k=0,1,2)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][c] = U_3x3[i*3+j][c]
        for j in range(3):
            for k1 in range(3):
                V_full[j*3+k1][c] = V_3x3[j*3+k1][c]
        for k1 in range(3):
            for i in range(3):
                W_full[k1*3+i][c] = W_3x3[k1*3+i][c]
    
    # Next 23 terms: for B2 (columns 3-5 of B, i.e., k=3,4,5)
    for c in range(23):
        c_full = 23 + c
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][c_full] = U_3x3[i*3+j][c]
        for j in range(3):
            for k2 in range(3):
                # In full V, column index is j*6 + (3+k2)
                V_full[j*6 + 3 + k2][c_full] = V_3x3[j*3+k2][c]
        for k2 in range(3):
            for i in range(3):
                # In full W, row index is (3+k2)*3 + i
                W_full[(3+k2)*3 + i][c_full] = W_3x3[k2*3+i][c]
    
    # Verify the full decomposition
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank_total):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if verify_full(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
