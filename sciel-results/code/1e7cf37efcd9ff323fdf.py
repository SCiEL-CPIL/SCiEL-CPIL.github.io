import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 blocks * 7 Strassen terms = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(6):
        # Strassen U factors (same for all blocks)
        # t=0: A00+A11
        U[0][col] = 1; U[3][col] = 1; col += 1
        # t=1: A10+A11
        U[2][col] = 1; U[3][col] = 1; col += 1
        # t=2: A00
        U[0][col] = 1; col += 1
        # t=3: A11
        U[3][col] = 1; col += 1
        # t=4: A00+A01
        U[0][col] = 1; U[1][col] = 1; col += 1
        # t=5: A10-A00
        U[2][col] = 1; U[0][col] = -1; col += 1
        # t=6: A01-A11
        U[1][col] = 1; U[3][col] = -1; col += 1
        
        # Strassen V factors for block b
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # t=0: B0k0 + B1k1
        V[0 * p + k0][col-7] = 1
        V[1 * p + k1][col-7] = 1
        # t=1: B0k0
        V[0 * p + k0][col-6] = 1
        # t=2: B1k0 - B1k1
        V[1 * p + k0][col-5] = 1
        V[1 * p + k1][col-5] = -1
        # t=3: B0k1 - B0k0
        V[0 * p + k1][col-4] = 1
        V[0 * p + k0][col-4] = -1
        # t=4: B1k1
        V[1 * p + k1][col-3] = 1
        # t=5: B0k0 + B0k1
        V[0 * p + k0][col-2] = 1
        V[0 * p + k1][col-2] = 1
        # t=6: B1k0 + B1k1
        V[1 * p + k0][col-1] = 1
        V[1 * p + k1][col-1] = 1
        
        # Strassen W factors for block b
        # Indices: k*n + i
        # k0=2b, k1=2b+1
        # i=0, i=1
        
        # t=0: C0k0 += 1, C1k1 += 1
        W[k0 * n + 0][col-7] = 1
        W[k1 * n + 1][col-7] = 1
        # t=1: C1k0 += 1, C1k1 += -1
        W[k0 * n + 1][col-6] = 1
        W[k1 * n + 1][col-6] = -1
        # t=2: C0k1 += 1, C1k0 += 1
        W[k1 * n + 0][col-5] = 1
        W[k0 * n + 1][col-5] = 1
        # t=3: C0k0 += 1, C0k1 += -1
        W[k0 * n + 0][col-4] = 1
        W[k1 * n + 0][col-4] = -1
        # t=4: C0k0 += -1, C0k1 += 1
        W[k0 * n + 0][col-3] = -1
        W[k1 * n + 0][col-3] = 1
        # t=5: C1k0 += -1, C1k1 += 1
        W[k0 * n + 1][col-2] = -1
        W[k1 * n + 1][col-2] = 1
        # t=6: C0k0 += 1, C1k0 += -1
        W[k0 * n + 0][col-1] = 1
        W[k0 * n + 1][col-1] = -1

    return {"U": U, "V": V, "W": W}
