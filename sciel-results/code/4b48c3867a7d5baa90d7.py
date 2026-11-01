import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block helper
    # i0, k0, j0 are the start indices
    # i1 = i0+1, k1 = k0+1, j1 = j0+1
    def add_strassen(i0, k0, j0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[i0*m + k0][col] += 1
        U[(i0+1)*m + (k0+1)][col] += 1
        V[k0*p + j0][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[j0*n + i0][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(i0+1)*m + k0][col] += 1
        U[(i0+1)*m + (k0+1)][col] += 1
        V[k0*p + j0][col] += 1
        W[j0*n + (i0+1)][col] += 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[i0*m + k0][col] += 1
        V[k0*p + (j0+1)][col] += 1
        V[(k0+1)*p + (j0+1)][col] -= 1
        W[(j0+1)*n + i0][col] += 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(i0+1)*m + (k0+1)][col] += 1
        V[(k0+1)*p + j0][col] += 1
        V[k0*p + j0][col] -= 1
        W[j0*n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[i0*m + k0][col] += 1
        U[i0*m + (k0+1)][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[(j0+1)*n + i0][col] -= 1
        W[(j0+1)*n + (i0+1)][col] += 1 # Wait, M5 contributes to C11 (-) and C22 (+)
        # Let's re-verify C formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # My W assignments above:
        # M5: W[j0+1, i0] -= 1 (C11 part), W[j0+1, i0+1] += 1 (C22 part).
        # This is correct.
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m + k0][col] += 1
        U[i0*m + k0][col] -= 1
        V[k0*p + j0][col] += 1
        V[k0*p + (j0+1)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1 # M6 contributes to C22
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i0*m + (k0+1)][col] += 1
        U[(i0+1)*m + (k0+1)][col] -= 1
        V[(k0+1)*p + j0][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[j0*n + i0][col] += 1 # M7 contributes to C11
        col += 1
        
    # Schoolbook 2x2x1 block helper
    # i0, k0, j0 (j is single)
    def add_schoolbook_2x2x1(i0, k0, j0):
        nonlocal col
        # M1 = A11 B11
        U[i0*m + k0][col] += 1
        V[k0*p + j0][col] += 1
        W[j0*n + i0][col] += 1
        col += 1
        
        # M2 = A12 B21
        U[i0*m + (k0+1)][col] += 1
        V[(k0+1)*p + j0][col] += 1
        W[j0*n + i0][col] += 1
        col += 1
        
        # M3 = A21 B11
        U[(i0+1)*m + k0][col] += 1
        V[k0*p + j0][col] += 1
        W[j0*n + (i0+1)][col] += 1
        col += 1
        
        # M4 = A22 B21
        U[(i0+1)*m + (k0+1)][col] += 1
        V[(k0+1)*p + j0][col] += 1
        W[j0*n + (i0+1)][col] += 1
        col += 1

    # Define blocks
    # k blocks: [0,1], [2,3]
    # j blocks: [0,1], [2,3], [4,5], [6]
    
    for k_start in [0, 2]:
        for j_start, j_len in [(0, 2), (2, 2), (4, 2), (6, 1)]:
            i_start = 0
            if j_len == 2:
                add_strassen(i_start, k_start, j_start)
            else:
                add_schoolbook_2x2x1(i_start, k_start, j_start)
                
    if col != rank:
        print(f"Warning: rank mismatch. Expected {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}
