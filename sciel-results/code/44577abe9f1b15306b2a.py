import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a schoolbook block
    # A_sub: rows r0..r1, cols c0..c1
    # B_sub: rows c0..c1, cols k0..k1
    # C_sub: rows r0..r1, cols k0..k1
    def add_schoolbook(r0, r1, c0, c1, k0, k1):
        nonlocal col_idx
        for i in range(r0, r1 + 1):
            for j in range(c0, c1 + 1):
                for k in range(k0, k1 + 1):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1

    # Helper to add a Strassen block for 2x2x2
    # A_sub: rows r0..r0+1, cols c0..c0+1
    # B_sub: rows c0..c0+1, cols k0..k0+1
    # C_sub: rows r0..r0+1, cols k0..k0+1
    def add_strassen(r0, c0, k0):
        nonlocal col_idx
        # Local indices in 2x2 block
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[r0 * m + c0][col_idx] = 1
        U[(r0+1) * m + (c0+1)][col_idx] = 1
        V[c0 * p + k0][col_idx] = 1
        V[(c0+1) * p + (k0+1)][col_idx] = 1
        W[k0 * n + r0][col_idx] = 1
        W[(k0+1) * n + (r0+1)][col_idx] = 1
        col_idx += 1
        
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        U[(r0+1) * m + c0][col_idx] = 1
        U[(r0+1) * m + (c0+1)][col_idx] = 1
        V[c0 * p + k0][col_idx] = 1
        W[(k0+1) * n + r0][col_idx] = 1
        W[k0 * n + (r0+1)][col_idx] = -1
        col_idx += 1
        
        # M3 = A00(B10-B11) -> C01+=1, C11+=1
        U[r0 * m + c0][col_idx] = 1
        V[(c0+1) * p + k0][col_idx] = 1
        V[(c0+1) * p + (k0+1)][col_idx] = -1
        W[k0 * n + (r0+1)][col_idx] = 1
        W[(k0+1) * n + (r0+1)][col_idx] = 1
        col_idx += 1
        
        # M4 = A11(B01-B10) -> C00+=1, C10+=1
        U[(r0+1) * m + (c0+1)][col_idx] = 1
        V[c0 * p + (k0+1)][col_idx] = 1
        V[(c0+1) * p + k0][col_idx] = -1
        W[k0 * n + r0][col_idx] = 1
        W[(k0+1) * n + r0][col_idx] = 1
        col_idx += 1
        
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        U[r0 * m + c0][col_idx] = 1
        U[r0 * m + (c0+1)][col_idx] = 1
        V[(c0+1) * p + (k0+1)][col_idx] = 1
        W[k0 * n + r0][col_idx] = -1
        W[k0 * n + (r0+1)][col_idx] = 1
        col_idx += 1
        
        # M6 = (A10-A00)(B00+B10) -> C11+=1
        U[(r0+1) * m + c0][col_idx] = 1
        U[r0 * m + c0][col_idx] = -1
        V[c0 * p + k0][col_idx] = 1
        V[(c0+1) * p + k0][col_idx] = 1
        W[(k0+1) * n + (r0+1)][col_idx] = 1
        col_idx += 1
        
        # M7 = (A01-A11)(B11+B01) -> C00+=1
        U[r0 * m + (c0+1)][col_idx] = 1
        U[(r0+1) * m + (c0+1)][col_idx] = -1
        V[(c0+1) * p + (k0+1)][col_idx] = 1
        V[c0 * p + (k0+1)][col_idx] = 1
        W[k0 * n + r0][col_idx] = 1
        col_idx += 1

    # Block definitions:
    # A rows: 0-1 (top), 2 (bot)
    # A cols: 0-1 (c0), 2-3 (c1), 4 (c2)
    # B rows: 0-1 (c0), 2-3 (c1), 4 (c2)
    # B cols: 0-1 (k0), 2-3 (k1), 4-5 (k2)
    
    # Top-Left (C rows 0-1, cols 0-1)
    add_strassen(0, 0, 0)       # A[0-1, 0-1] * B[0-1, 0-1]
    add_strassen(0, 2, 0)       # A[0-1, 2-3] * B[2-3, 0-1]
    add_schoolbook(0, 1, 4, 4, 0, 1) # A[0-1, 4] * B[4, 0-1]
    
    # Top-Mid (C rows 0-1, cols 2-3)
    add_strassen(0, 0, 2)       # A[0-1, 0-1] * B[0-1, 2-3]
    add_strassen(0, 2, 2)       # A[0-1, 2-3] * B[2-3, 2-3]
    add_schoolbook(0, 1, 4, 4, 2, 3) # A[0-1, 4] * B[4, 2-3]
    
    # Top-Right (C rows 0-1, cols 4-5)
    add_strassen(0, 0, 4)       # A[0-1, 0-1] * B[0-1, 4-5]
    add_strassen(0, 2, 4)       # A[0-1, 2-3] * B[2-3, 4-5]
    add_schoolbook(0, 1, 4, 4, 4, 5) # A[0-1, 4] * B[4, 4-5]
    
    # Bot-Left (C row 2, cols 0-1)
    add_schoolbook(2, 2, 0, 1, 0, 1) # A[2, 0-1] * B[0-1, 0-1]
    add_schoolbook(2, 2, 2, 3, 0, 1) # A[2, 2-3] * B[2-3, 0-1]
    add_schoolbook(2, 2, 4, 4, 0, 1) # A[2, 4] * B[4, 0-1]
    
    # Bot-Mid (C row 2, cols 2-3)
    add_schoolbook(2, 2, 0, 1, 2, 3) # A[2, 0-1] * B[0-1, 2-3]
    add_schoolbook(2, 2, 2, 3, 2, 3) # A[2, 2-3] * B[2-3, 2-3]
    add_schoolbook(2, 2, 4, 4, 2, 3) # A[2, 4] * B[4, 2-3]
    
    # Bot-Right (C row 2, cols 4-5)
    add_schoolbook(2, 2, 0, 1, 4, 5) # A[2, 0-1] * B[0-1, 4-5]
    add_schoolbook(2, 2, 2, 3, 4, 5) # A[2, 2-3] * B[2-3, 4-5]
    add_schoolbook(2, 2, 4, 4, 4, 5) # A[2, 4] * B[4, 4-5]
    
    return {"U": U, "V": V, "W": W}
