import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 block: multiplies 2x2 A block by 2x2 B block
    # Uses 7 multiplications instead of 4
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        # Input indices (A matrix is n x m = 2 x 4)
        a = i0 * m + j0          # A[i0][j0]
        b = i0 * m + (j0 + 1)    # A[i0][j0+1]
        c = (i0 + 1) * m + j0    # A[i0+1][j0]
        d = (i0 + 1) * m + (j0 + 1)  # A[i0+1][j0+1]
        
        # B matrix is m x p = 4 x 7
        e = j0 * p + k0          # B[j0][k0]
        f = j0 * p + (k0 + 1)    # B[j0][k0+1]
        g = (j0 + 1) * p + k0    # B[j0+1][k0]
        h = (j0 + 1) * p + (k0 + 1)  # B[j0+1][k0+1]
        
        # Output indices (C matrix is n x p = 2 x 7, stored as C[k*n+i])
        c00 = k0 * n + i0        # C[i0][k0]
        c01 = (k0 + 1) * n + i0  # C[i0][k0+1]
        c10 = k0 * n + (i0 + 1)  # C[i0+1][k0]
        c11 = (k0 + 1) * n + (i0 + 1)  # C[i0+1][k0+1]
        
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = A22(B21 - B11)
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = (A11 - A21)B22
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = A11(B12 - B22)
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A12 + A11)B11
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A21 + A11)(B12 + B22)
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True
    
    # Schoolbook 2x2x1 block: multiplies 2x2 A block by 2x1 B block
    # Uses 4 multiplications
    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # C[i0][k0] = A[i0][j0]*B[j0][k0] + A[i0][j0+1]*B[j0+1][k0]
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        # C[i0+1][k0] = A[i0+1][j0]*B[j0][k0] + A[i0+1][j0+1]*B[j0+1][k0]
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Process k-blocks for j=0 (columns 0-1 of B)
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Handle k=6 separately with schoolbook (single column)
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Process k-blocks for j=2 (columns 2-3 of B)
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Handle k=6 separately with schoolbook
    if not add_schoolbook_221(0, 2, 6):
        pass

    return {"U": U, "V": V, "W": W}
