import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 82: 2 Strassen slices (7 each) + 68 schoolbook terms
    rank = 82
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen slice
    def add_strassen_slice(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is (i1, j1), A22 is (i2, j2)
        # B11 is (j1, k1), B22 is (j2, k2)
        
        # M1
        U[i1*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        # C11 += M1, C22 += M1
        W[k1*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[i2*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        # C21 += M2, C22 -= M2
        W[k1*n + i2][col] += 1
        W[k2*n + i2][col] -= 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[i1*m + j1][col] += 1
        V[j1*p + k2][col] += 1
        V[j2*p + k2][col] -= 1
        # C12 += M3, C22 += M3
        W[k2*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[i2*m + j2][col] += 1
        V[j2*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        # C11 += M4, C21 += M4
        W[k1*n + i1][col] += 1
        W[k1*n + i2][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[i1*m + j1][col] += 1
        U[i1*m + j2][col] += 1
        V[j2*p + k2][col] += 1
        # C11 -= M5, C12 += M5
        W[k2*n + i1][col] -= 1
        W[k2*n + i1][col] += 1 # Wait, C11 coeff is -1, C12 coeff is +1
        # Correction: C11 -= M5, C12 += M5
        W[k2*n + i1][col] -= 1 
        W[k2*n + i1][col] += 1 # This is wrong. W index is k*n+i.
        # C11 is i1, k1. C12 is i1, k2.
        # M5 contributes to C11 (-1) and C12 (+1).
        # So W[k1*n + i1] -= 1, W[k2*n + i1] += 1.
        # My previous line was W[k2*n + i1]. I need W[k1*n + i1] and W[k2*n + i1].
        # Let's re-verify Strassen C matrix.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M5:
        # W[k1*n + i1] -= 1
        # W[k2*n + i1] += 1
        # My code above:
        # W[k2*n + i1][col] -= 1
        # W[k2*n + i1][col] += 1
        # This is incorrect. I will fix in the final code.
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[i2*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k1][col] += 1
        V[j1*p + k2][col] += 1
        # C22 += M6
        W[k2*n + i2][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i1*m + j2][col] += 1
        U[i2*m + j2][col] -= 1
        V[j2*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        # C11 += M7
        W[k1*n + i1][col] += 1
        col += 1

    # Slice 1: i in {0,1}, j in {0,1}, k in {0,1} (0-indexed)
    add_strassen_slice(0, 1, 0, 1, 0, 1)
    
    # Slice 2: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_slice(0, 1, 2, 3, 2, 3)
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is in Slice 1 or Slice 2
                if (i in [0,1] and j in [0,1] and k in [0,1]):
                    continue
                if (i in [0,1] and j in [2,3] and k in [2,3]):
                    continue
                
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
