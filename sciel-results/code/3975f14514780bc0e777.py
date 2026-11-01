import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # rank = 7 * 7 + 4 = 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 pairs of columns: (0,1), (2,3), ..., (12,13)
    for pair_idx in range(7):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # Strassen terms for block k0, k1
        # M1
        U[0][col] += 1 # A00
        U[3][col] += 1 # A11
        V[15*k0 + 0][col] += 1 # B00 (j=0, k=k0) -> V[0*15+k0]
        V[15*k1 + 1][col] += 1 # B11 (j=1, k=k1) -> V[1*15+k1]
        # W for C00, C01, C10, C11 contributions
        # C00 = M1 + ...
        W[k0*n + 0][col] += 1
        # C11 = M1 + ...
        W[k1*n + 1][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1 # A10
        U[3][col] += 1 # A11
        V[15*k0 + 0][col] += 1 # B00
        # C10 = M2 + ...
        W[k0*n + 1][col] += 1
        # C11 = ... - M2
        W[k1*n + 1][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1 # A00
        V[15*k1 + 1][col] += 1 # B11
        V[15*k0 + 0][col] -= 1 # -B00
        # C01 = M3 + ...
        W[k1*n + 0][col] += 1
        # C11 = ... + M3
        W[k1*n + 1][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1 # A11
        V[15*k0 + 0][col] += 1 # B00
        V[15*k1 + 1][col] -= 1 # -B11
        # C00 = ... + M4
        W[k0*n + 0][col] += 1
        # C10 = ... + M4
        W[k0*n + 1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1 # A00
        U[1][col] += 1 # A01
        V[15*k0 + 1][col] += 1 # B10 (j=1, k=k0) -> V[1*15+k0]
        # C00 = ... - M5
        W[k0*n + 0][col] -= 1
        # C01 = ... + M5
        W[k1*n + 0][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1 # A10
        U[0][col] -= 1 # -A00
        V[15*k0 + 0][col] += 1 # B00
        V[15*k1 + 0][col] += 1 # B01 (j=0, k=k1) -> V[0*15+k1]
        # C11 = ... + M6
        W[k1*n + 1][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1 # A01
        U[3][col] -= 1 # -A11
        V[15*k0 + 1][col] += 1 # B10
        V[15*k1 + 1][col] += 1 # B11
        # C00 = ... + M7
        W[k0*n + 0][col] += 1
        col += 1

    # 1 single column: k=14
    k = 14
    # Schoolbook for single column
    # C00 = A00 B00
    U[0][col] = 1
    V[15*k + 0][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    # C01 = A01 B10? No, C_{0, k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
    # So for single column k:
    # C_{0,k} = A_{00}B_{0k} + A_{01}B_{1k}
    # C_{1,k} = A_{10}B_{0k} + A_{11}B_{1k}
    
    # Term 1: A00 B0k -> C0k
    U[0][col] = 1
    V[15*k + 0][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # Term 2: A01 B1k -> C0k
    U[1][col] = 1
    V[15*k + 1][col] = 1
    W[k*n + 0][col] = 1
    col += 1
    
    # Term 3: A10 B0k -> C1k
    U[2][col] = 1
    V[15*k + 0][col] = 1
    W[k*n + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1k -> C1k
    U[3][col] = 1
    V[15*k + 1][col] = 1
    W[k*n + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
