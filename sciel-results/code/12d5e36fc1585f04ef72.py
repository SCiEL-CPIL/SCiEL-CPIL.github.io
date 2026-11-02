import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = 150.0

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    # Test with random matrices
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute AB directly
    C_direct = [[0]*p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C_direct[i][k] = s
    
    # Compute using decomposition
    C_decomp = [[0]*p for _ in range(n)]
    for c in range(r):
        # Compute linear forms
        a_form = [0]*n
        for i in range(n):
            for j in range(m):
                a_form[i] += U[i*m+j][c] * A[i][j]
        
        b_form = [0]*p
        for k in range(p):
            for j in range(m):
                b_form[k] += V[j*p+k][c] * B[j][k]
        
        for i in range(n):
            for k in range(p):
                C_decomp[i][k] += a_form[i] * b_form[k] * W[k*n+i][c]
    
    for i in range(n):
        for k in range(p):
            if C_direct[i][k] != C_decomp[i][k]:
                return False
    return True

def construct():
    n, m, p = 2, 3, 8
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 6 x 36
    V = [[0]*rank for _ in range(m*p)]  # 24 x 36
    W = [[0]*rank for _ in range(p*n)]  # 16 x 36
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> 2x4 outer product (8 terms)
    for i in range(2):
        for k in range(4):
            U[i*m + 0][col] = 1      # A[i,0]
            V[0*p + k][col] = 1       # B[0,k]
            W[k*n + i][col] = 1       # C[i,k]
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x2 x 2x4
    # Use Strassen on 4 pairs of columns
    # A'[i,0] = A[i,1], A'[i,1] = A[i,2]
    # B'[0,k] = B[1,k], B'[1,k] = B[2,k]
    
    # Strassen 2x2x2 coefficients:
    # Need to find correct formulas. Let me use a known correct set.
    # After research, the correct Strassen is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 - M3 + M6
    
    # But my verification showed this doesn't work. Let me try another set.
    # Actually, I think the issue is that I need to be more careful.
    
    # Let me just use schoolbook for the 2x2x4 part to be safe.
    # That would be 2*2*4 = 16 terms, total 8+16 = 24... wait that's less than 36!
    
    # Actually, for 2x2 x 2x4, schoolbook is 2*2*4 = 16 terms.
    # Total: 8 + 16 = 24 terms. That's much better than 48!
    
    # Let me redo this with schoolbook for the second part.
    rank = 24
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Part 1: A[:,0] * B[0,:] (8 terms)
    for i in range(2):
        for k in range(4):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:] (16 terms, schoolbook)
    for i in range(2):
        for j in range(2):  # j=0,1 correspond to original j=1,2
            for k in range(4):
                U[i*m + (j+1)][col] = 1   # A[i, j+1]
                V[(j+1)*p + k][col] = 1    # B[j+1, k]
                W[k*n + i][col] = 1         # C[i,k]
                col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
