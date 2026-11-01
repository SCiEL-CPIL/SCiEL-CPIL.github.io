import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen decomposition for 2x2x2 blocks
    # 5 blocks of 2 columns (0-9)
    for b in range(5):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1: (A00 + A11) * (B0k1 + B1k2) -> C0k1, C1k2
        U[0, col] = 1; U[3, col] = 1
        V[k1, col] = 1; V[m + k2, col] = 1
        W[k1 * n + 0, col] = 1; W[k2 * n + 1, col] = 1
        col += 1
        
        # M2: (A10 + A11) * B0k1 -> C1k1
        U[2, col] = 1; U[3, col] = 1
        V[k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M3: A00 * (B0k2 - B1k2) -> C0k2, C1k2
        U[0, col] = 1
        V[k2, col] = 1; V[m + k2, col] = -1
        W[k2 * n + 0, col] = 1; W[k2 * n + 1, col] = 1
        col += 1
        
        # M4: A11 * (B1k1 - B0k1) -> C0k1, C1k1
        U[3, col] = 1
        V[m + k1, col] = 1; V[k1, col] = -1
        W[k1 * n + 0, col] = 1; W[k1 * n + 1, col] = 1
        col += 1
        
        # M5: (A00 + A01) * B1k2 -> C0k2, -C0k1
        U[0, col] = 1; U[1, col] = 1
        V[m + k2, col] = 1
        W[k2 * n + 0, col] = 1; W[k1 * n + 0, col] = -1
        col += 1
        
        # M6: (A10 - A00) * (B0k1 + B0k2) -> C1k2
        U[2, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[k2, col] = 1
        W[k2 * n + 1, col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B1k1 + B1k2) -> C0k1
        U[1, col] = 1; U[3, col] = -1
        V[m + k1, col] = 1; V[m + k2, col] = 1
        W[k1 * n + 0, col] = 1
        col += 1

    # Last column (10) - trivial 2x2x1
    k = 10
    # A00 * B0k -> C0k
    U[0, col] = 1
    V[k, col] = 1
    W[k * n + 0, col] = 1
    col += 1
    
    # A01 * B1k -> C0k
    U[1, col] = 1
    V[m + k, col] = 1
    W[k * n + 0, col] = 1
    col += 1
    
    # A10 * B0k -> C1k
    U[2, col] = 1
    V[k, col] = 1
    W[k * n + 1, col] = 1
    col += 1
    
    # A11 * B1k -> C1k
    U[3, col] = 1
    V[m + k, col] = 1
    W[k * n + 1, col] = 1
    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
