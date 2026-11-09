import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 27 * 7 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # Indices for U2: (a, b) where a is row, b is col of A
    # Indices for V2: (b, c) where b is row, c is col of B
    # Indices for W2: (c, a) where c is row, a is col of C (output)
    # Note: W row index is k*n + i. Here k=c, i=a.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[3][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6
    U2[2][5] = 1; U2[3][5] = -1
    V2[2][5] = 1; V2[3][5] = 1
    W2[3][5] = 1
    
    # M7
    U2[0][6] = 1; U2[2][6] = -1
    V2[0][6] = 1; V2[1][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Schoolbook factors for 3x3
    # r3 = 9*i0 + 3*j0 + k0
    # U3[(i1, j1), r3] = 1 if i1=i0, j1=j0
    # V3[(j1, k1), r3] = 1 if j1=j0, k1=k0
    # W3[(k1, i1), r3] = 1 if k1=k0, i1=i0
    
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                r3 = 9 * i1 + 3 * j1 + k1
                for r2 in range(7):
                    r = 7 * r3 + r2
                    u_val = U2[2*i1 + 0, r2] # Wait, U2 index is (a,b)
                    # U row index is i*m + j = (2*i1+a)*6 + (2*j1+b) = 12*i1 + 6*a + 2*j1 + b
                    # This is not a simple Kronecker product of indices if we just multiply values.
                    # We need to map (i1, a, j1, b) to the row index.
                    # Row index of U: i*m + j = (2*i1+a)*6 + (2*j1+b)
                    
                    # Let's iterate over all i, j, k to fill U, V, W
                    # This is O(216 * 189) which is small.
                    
    # Better approach: Iterate over r3, r2 and fill the specific non-zero entries.
    # But we already initialized to 0. We can just add the products.
    # Since U, V, W are sparse in the block sense, we can iterate over i1, j1, k1, a, b, c.
    
    # Let's reset and fill properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                r3 = 9 * i1 + 3 * j1 + k1
                # Schoolbook part is 1 for this r3
                for r2 in range(7):
                    r = 7 * r3 + r2
                    # U2, V2, W2 values for r2
                    for a in range(2):
                        for b in range(2):
                            u2_val = U2[a*2+b][r2]
                            if u2_val != 0:
                                i = 2*i1 + a
                                j = 2*j1 + b
                                row_u = i*m + j
                                U[row_u][r] += u2_val
                    for b in range(2):
                        for c in range(2):
                            v2_val = V2[b*2+c][r2]
                            if v2_val != 0:
                                j = 2*j1 + b
                                k = 2*k1 + c
                                row_v = j*p + k
                                V[row_v][r] += v2_val
                    for c in range(2):
                        for a in range(2):
                            w2_val = W2[c*2+a][r2]
                            if w2_val != 0:
                                k = 2*k1 + c
                                i = 2*i1 + a
                                row_w = k*n + i
                                W[row_w][r] += w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
