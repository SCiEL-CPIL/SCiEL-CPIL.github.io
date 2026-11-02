"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2."""
    # Strassen's 7 multiplication formulas for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # The 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, we need to express this as:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # where i,j,k are indices into the flattened matrices.
    #
    # Actually, the correct formulation is that the matrix multiplication tensor
    # T[i,j,k] (where i indexes rows of A, j indexes columns of A/rows of B, 
    # k indexes columns of B) has the property that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor decomposition means we can write the multiplication map as:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta_{i,j,k} (in some sense)
    #
    # Let me use the standard Strassen formulation more carefully.
    # 
    # For 2x2x2, the schoolbook has rank 8. Strassen reduces it to 7.
    # The 7 terms correspond to 7 bilinear forms.
    
    rank = 7
    U = [[0] * rank for _ in range(4)]  # n*m = 2*2 = 4
    V = [[0] * rank for _ in range(4)]  # m*p = 2*2 = 4
    W = [[0] * rank for _ in range(4)]  # p*n = 2*2 = 4
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j,k in {0,1}
    
    # M1 = (a+d)(e+h): 
    # a = A[0,0] = U[0*2+0, r]
    # d = A[1,1] = U[1*2+1, r]
    # e = B[0,0] = V[0*2+0, r]
    # h = B[1,1] = V[1*2+1, r]
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    # W for M1 contributes to C[0,0] and C[1,1]
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # M2 = (c+d)e:
    # c = A[1,0] = U[1*2+0, r] = U[2, r]
    # d = A[1,1] = U[3, r]
    # e = B[0,0] = V[0, r]
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1 # C[1,1]
    
    # M3 = a(f-h):
    # a = A[0,0] = U[0, r]
    # f = B[0,1] = V[1, r]
    # h = B[1,1] = V[3, r]
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # -h
    # C12 = M3 + M5
    # C22 = M1 - M2 + M3 + M6
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # M4 = d(g-e):
    # d = A[1,1] = U[3, r]
    # g = B[1,0] = V[2, r]
    # e = B[0,0] = V[0, r]
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1 # -e
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M4
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # M5 = (a+b)h:
    # a = A[0,0] = U[0, r]
    # b = A[0,1] = U[1, r]
    # h = B[1,1] = V[3, r]
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    W[0][4] = -1 # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # M6 = (c-a)(e+f):
    # c = A[1,0] = U[2, r]
    # a = A[0,0] = U[0, r]
    # e = B[0,0] = V[0, r]
    # f = B[0,1] = V[1, r]
    U[2][5] = 1  # c
    U[0][5] = -1 # -a
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    # C22 = M1 - M2 + M3 + M6
    W[3][5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h):
    # b = A[0,1] = U[1, r]
    # d = A[1,1] = U[3, r]
    # g = B[1,0] = V[2, r]
    # h = B[1,1] = V[3, r]
    U[1][6] = 1  # b
    U[3][6] = -1 # -d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    # C11 = M1 + M4 - M5 + M7
    W[0][6] = 1  # C[0,0]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_n(n1, n2, m, p, r1, r2):
    """Combine decompositions split on first dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range((n1+n2) * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * (n1+n2))]
    
    for i in range(n1 * m):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n1):
            for r in range(r1['rank']):
                W[k*(n1+n2)+i][r] = r1['W'][k*n1+i][r]
    
    off = r1['rank']
    for i in range(n2 * m):
        for r in range(r2['rank']):
            U[n1*m + i][off+r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p):
            for r in range(r2['rank']):
                V[j*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n2):
            for r in range(r2['rank']):
                W[k*(n1+n2)+n1+i][off+r] = r2['W'][k*n2+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_m(n, m1, m2, p, r1, r2):
    """Combine decompositions split on middle dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * (m1+m2))]
    V = [[0] * rank for _ in range((m1+m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m1):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m1):
        for k in range(p):
            for r in range(r1['rank']):
                V[j*p+k][r] = r1['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
    
    off = r1['rank']
    for i in range(n * m2):
        for r in range(r2['rank']):
            U[n*m1 + i][off+r] = r2['U'][i][r]
    for j in range(m2):
        for k in range(p):
            for r in range(r2['rank']):
                V[(m1+j)*p+k][off+r] = r2['V'][j*p+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(r2['rank']):
                W[k*n+i][off+r] = r2['W'][k*n+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_p(n, m, p1, p2, r1, r2):
    """Combine decompositions split on third dimension."""
    rank = r1['rank'] + r2['rank']
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * (p1+p2))]
    W = [[0] * rank for _ in range((p1+p2) * n)]
    
    for i in range(n * m):
        for r in range(r1['rank']):
            U[i][r] = r1['U'][i][r]
    for j in range(m):
        for k in range(p1):
            for r in range(r1['rank']):
                V[j*(p1+p2)+k][r] = r1['V'][j*p1+k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(r1['rank']):
                W[k*n+i][r] = r1['W'][k*n+i][r]
    
    off = r1['rank']
    for i in range(n * m):
        for r in range(r2['rank']):
            U[i][off+r] = r2['U'][i][r]
    for j in range(m):
        for k in range(p2):
            for r in range(r2['rank']):
                V[j*(p1+p2)+p1+k][off+r] = r2['V'][j*p2+k][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2['rank']):
                W[(p1+k)*n+i][off+r] = r2['W'][k*n+i][r]
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _best_decompose(n, m, p, deadline=None):
    """Find the best known decomposition for n×m × m×p using block decomposition."""
    if deadline is not None and time.time() > deadline:
        return _schoolbook(n, m, p)
    
    if n == 0 or m == 0 or p == 0:
        return {'U': [], 'V': [], 'W': [], 'rank': 0}
    
    # Base case: 2x2x2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Base case: if any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
    
    best = _schoolbook(n, m, p)
    best_rank = best['rank']
    
    # Try splitting n
    for pos in range(1, n):
        if deadline is not None and time.time() > deadline:
            break
        n1, n2 = pos, n - pos
        r1 = _best_decompose(n1, m, p, deadline)
        r2 = _best_decompose(n2, m, p, deadline)
        combined = _combine_n(n1, n2, m, p, r1, r2)
        if combined['rank'] < best_rank:
            best = combined
            best_rank = combined['rank']
    
    # Try splitting m
    for pos in range(1, m):
        if deadline is not None and time.time() > deadline:
            break
        m1, m2 = pos, m - pos
        r1 = _best_decompose(n, m1, p, deadline)
        r2 = _best_decompose(n, m2, p, deadline)
        combined = _combine_m(n, m1, m2, p, r1, r2)
        if combined['rank'] < best_rank:
            best = combined
            best_rank = combined['rank']
    
    # Try splitting p
    for pos in range(1, p):
        if deadline is not None and time.time() > deadline:
            break
        p1, p2 = pos, p - pos
        r1 = _best_decompose(n, m, p1, deadline)
        r2 = _best_decompose(n, m, p2, deadline)
        combined = _combine_p(n, m, p1, p2, r1, r2)
        if combined['rank'] < best_rank:
            best = combined
            best_rank = combined['rank']
    
    return best

def construct():
    n, m, p = DIMENSIONS
    
    # Set a deadline with safety margin (150s limit, use 140s)
    deadline = time.time() + 140
    
    # Try the recursive block decomposition with Strassen base case
    result = _best_decompose(n, m, p, deadline)
    
    return {"U": result['U'], "V": result['V'], "W": result['W']}
# EVOLVE-BLOCK-END
