import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 5 blocks of 2 columns each
    for b in range(5):
        r_base = b * 11
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen Part (7 terms) ---
        # Indices in U: i * m + j. i in 0..1, j in 0..1 for Strassen part.
        # Indices in V: j * p + k. j in 0..1, k in k0, k1 for Strassen part.
        # Indices in W: k * n + i.
        
        # M1: A[0,0] * (B[0,k0] + B[1,k1])
        r = r_base + 0
        U[0 * m + 0, r] = 1
        V[0 * p + k0, r] = 1
        V[1 * p + k1, r] = 1
        W[k0 * n + 0, r] = 1   # C[0, k0]
        W[k1 * n + 1, r] = 1   # C[1, k1]
        
        # M2: (A[1,0] + A[1,1]) * B[1,k1]
        r = r_base + 1
        U[1 * m + 0, r] = 1
        U[1 * m + 1, r] = 1
        V[1 * p + k1, r] = 1
        W[k0 * n + 1, r] = 1   # C[1, k0]
        
        # M3: (A[0,0] + A[0,1]) * B[0,k0]
        r = r_base + 2
        U[0 * m + 0, r] = 1
        U[0 * m + 1, r] = 1
        V[0 * p + k0, r] = 1
        W[k1 * n + 0, r] = 1   # C[0, k1]
        W[k1 * n + 1, r] = -1  # C[1, k1]
        
        # M4: A[1,1] * (B[1,k0] + B[0,k0])
        r = r_base + 3
        U[1 * m + 1, r] = 1
        V[1 * p + k0, r] = 1
        V[0 * p + k0, r] = 1
        W[k0 * n + 0, r] = -1  # C[0, k0]
        W[k1 * n + 1, r] = 1   # C[1, k1]
        
        # M5: (A[0,1] + A[1,1]) * (B[1,k0] + B[1,k1])
        r = r_base + 4
        U[0 * m + 1, r] = 1
        U[1 * m + 1, r] = 1
        V[1 * p + k0, r] = 1
        V[1 * p + k1, r] = 1
        W[k0 * n + 0, r] = 1   # C[0, k0]
        
        # M6: (A[0,0] - A[1,0]) * (B[0,k0] + B[1,k0])
        r = r_base + 5
        U[0 * m + 0, r] = 1
        U[1 * m + 0, r] = -1
        V[0 * p + k0, r] = 1
        V[1 * p + k0, r] = 1
        W[k0 * n + 1, r] = 1   # C[1, k0]
        W[k1 * n + 1, r] = 1   # C[1, k1]
        
        # M7: (A[0,1] - A[1,1]) * (B[0,k1] + B[1,k1])
        r = r_base + 6
        U[0 * m + 1, r] = 1
        U[1 * m + 1, r] = -1
        V[0 * p + k1, r] = 1
        V[1 * p + k1, r] = 1
        W[k0 * n + 0, r] = -1  # C[0, k0]
        W[k1 * n + 0, r] = 1   # C[0, k1]
        
        # --- Direct Part (4 terms) for j=2 ---
        # C[i, k] += A[i, 2] * B[2, k]
        
        # i=0, k=k0
        r = r_base + 7
        U[0 * m + 2, r] = 1
        V[2 * p + k0, r] = 1
        W[k0 * n + 0, r] = 1
        
        # i=0, k=k1
        r = r_base + 8
        U[0 * m + 2, r] = 1
        V[2 * p + k1, r] = 1
        W[k1 * n + 0, r] = 1
        
        # i=1, k=k0
        r = r_base + 9
        U[1 * m + 2, r] = 1
        V[2 * p + k0, r] = 1
        W[k0 * n + 1, r] = 1
        
        # i=1, k=k1
        r = r_base + 10
        U[1 * m + 2, r] = 1
        V[2 * p + k1, r] = 1
        W[k1 * n + 1, r] = 1

    return {'U': U, 'V': V, 'W': W}
