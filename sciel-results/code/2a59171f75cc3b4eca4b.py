import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook terms for remaining pairs
    # Pairs: (0,2), (0,3), (0,4), (1,2), (2,0), (2,1), (2,2)
    # Wait, I need to list all pairs NOT in S1 or S2.
    # S1: j in {0,1}, k in {0,1}
    # S2: j in {1,2}, k in {3,4}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j in [0, 1] and k in [0, 1]) or (j in [1, 2] and k in [3, 4]):
                    continue
                # Schoolbook
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Strassen S1: j in {0,1}, k in {0,1}
    # Map j' = j, k' = k
    # Coeffs for M1..M7
    # alpha[i, j], beta[j, k], gamma[i, k]
    
    def add_strassen(j_offset, k_offset):
        nonlocal col
        # M1
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + 0 + k_offset][col] = 1
        V[1 * p + 1 + k_offset][col] = 1 # Wait, V index is j*p + k
        # Correction: V[j*p+k]
        # j is local 0,1. Global j = j + j_offset
        # k is local 0,1. Global k = k + k_offset
        
        # Let's rewrite add_strassen to be safe
        # U indices: i*m + (j + j_offset)
        # V indices: (j + j_offset)*p + (k + k_offset)
        # W indices: (k + k_offset)*n + i
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0 * m + (0 + j_offset)][col] += 1
        U[1 * m + (1 + j_offset)][col] += 1
        V[(0 + j_offset) * p + (0 + k_offset)][col] += 1
        V[(1 + j_offset) * p + (1 + k_offset)][col] += 1
        W[(0 + k_offset) * n + 0][col] += 1
        W[(1 + k_offset) * n + 1][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[1 * m + (0 + j_offset)][col] += 1
        U[1 * m + (1 + j_offset)][col] += 1
        V[(0 + j_offset) * p + (0 + k_offset)][col] += 1
        W[(0 + k_offset) * n + 1][col] += 1
        W[(1 + k_offset) * n + 1][col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[0 * m + (0 + j_offset)][col] += 1
        V[(0 + j_offset) * p + (1 + k_offset)][col] += 1
        V[(1 + j_offset) * p + (1 + k_offset)][col] += -1
        W[(1 + k_offset) * n + 0][col] += 1
        W[(1 + k_offset) * n + 1][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[1 * m + (1 + j_offset)][col] += 1
        V[(1 + j_offset) * p + (0 + k_offset)][col] += 1
        V[(0 + j_offset) * p + (0 + k_offset)][col] += -1
        W[(0 + k_offset) * n + 0][col] += 1
        W[(0 + k_offset) * n + 1][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[0 * m + (0 + j_offset)][col] += 1
        U[0 * m + (1 + j_offset)][col] += 1
        V[(1 + j_offset) * p + (1 + k_offset)][col] += 1
        W[(0 + k_offset) * n + 0][col] += -1
        W[(1 + k_offset) * n + 0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> -C01, C10
        U[1 * m + (0 + j_offset)][col] += 1
        U[0 * m + (0 + j_offset)][col] += -1
        V[(0 + j_offset) * p + (0 + k_offset)][col] += 1
        V[(0 + j_offset) * p + (1 + k_offset)][col] += 1
        W[(1 + k_offset) * n + 0][col] += -1
        W[(0 + k_offset) * n + 1][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> -C00, C11
        U[0 * m + (1 + j_offset)][col] += 1
        U[1 * m + (1 + j_offset)][col] += -1
        V[(1 + j_offset) * p + (0 + k_offset)][col] += 1
        V[(1 + j_offset) * p + (1 + k_offset)][col] += 1
        W[(0 + k_offset) * n + 0][col] += -1
        W[(1 + k_offset) * n + 1][col] += 1
        col += 1

    add_strassen(0, 0)
    add_strassen(1, 3)
    
    return {"U": U, "V": V, "W": W}
