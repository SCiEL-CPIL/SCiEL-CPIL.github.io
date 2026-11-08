import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 26 solution
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for top-left 2x2 block (indices 0,1,2,3 for A and B, 0,1,2,3 for C)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # Indices in 3x3:
    # A: 0,1,3,4 (a,b,c,d) -> 0,1,3,4
    # B: 0,1,3,4 (e,f,g,h) -> 0,1,3,4
    # C: 0,1,3,4 (i,j,k,l) -> 0,1,3,4
    
    # M1 = (c+d)(g+h) -> i = M1 - M2 - M4 + M7? No, standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)e
    # M6 = (c+a)f
    # M7 = (d-b)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 - M6
    
    # Mapping to 3x3 indices:
    # A[0]=a, A[1]=b, A[3]=c, A[4]=d
    # B[0]=e, B[1]=f, B[3]=g, B[4]=h
    # C[0]=i, C[1]=j, C[3]=k, C[4]=l
    
    # M1: (a+d)(e+h)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2: (c+d)h
    U[3][col] = 1; U[4][col] = 1
    V[4][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M3: a(f-h)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: d(g-e)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M5: (a+b)e
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (c+a)f
    U[3][col] = 1; U[0][col] = 1
    V[1][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M7: (d-b)(g+h)
    U[4][col] = 1; U[1][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = -1
    col += 1
    
    # Cross terms involving the 3rd row/col
    # A: [a b c; d e f; g h i] -> 0,1,2,3,4,5,6,7,8
    # B: [j k l; m n o; p q r] -> 0,1,2,3,4,5,6,7,8
    # C: [s t u; v w x; y z alpha] -> 0,1,2,3,4,5,6,7,8
    
    # The 2x2 Strassen handled A[0,1,3,4], B[0,1,3,4], C[0,1,3,4].
    # Remaining elements:
    # A: 2, 5, 6, 7, 8
    # B: 2, 5, 6, 7, 8
    # C: 2, 5, 6, 7, 8
    
    # We need to compute C[2], C[5], C[6], C[7], C[8] and contributions to C[0..4] from cross terms.
    # C[0] += A[0]*B[2] + A[1]*B[5] + A[2]*B[8] + A[3]*B[2] + A[4]*B[5] + A[5]*B[8]
    # This is getting complex. The seed code used a specific set of "add_term" calls.
    # I will replicate the seed's cross terms exactly.
    
    def add_term(u, v, w, c):
        nonlocal col
        U[u][col] += c
        V[v][col] = 1
        W[w][col] = 1
        col += 1
        
    # a b^T (2x1 * 1x2) - likely refers to specific sub-blocks
    # Seed code:
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # A2 b (2x2 * 2x1)
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # a b3 (2x1 * 1x1)
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # a^T B2 (1x2 * 2x2)
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # a3 b^T (1x1 * 1x2)
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # a^T b (1x2 * 2x1)
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # a3 b3
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
