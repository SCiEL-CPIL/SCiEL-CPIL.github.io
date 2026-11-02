import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # rank is 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2 columns
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Schoolbook part for A_R B_b,B
        # A_R is A[:, 2] (col index 2)
        # B_b,B is B[2, :] (row index 2)
        
        # 1. A[0,2] * B[2, k0]
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # 2. A[0,2] * B[2, k1]
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # 3. A[1,2] * B[2, k0]
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # 4. A[1,2] * B[2, k1]
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Strassen part for A_L B_b,T
        # A_L is A[:, 0:2]
        # B_b,T is B[0:2, :]
        
        # P1 = (A[0,0] + A[0,1]) * (B[0, k0] + B[1, k1])
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2 = (A[1,0] + A[1,1]) * B[0, k0]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3 = A[0,0] * (B[0, k1] - B[1, k1])
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4 = A[1,1] * (B[1, k0] - B[0, k0])
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5 = (A[0,0] + A[0,1]) * B[1, k1]
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6 = (A[1,0] - A[1,1]) * (B[0, k0] + B[0, k1])
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7 = (A[1,0] + A[1,1]) * B[1, k0]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
