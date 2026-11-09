import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # 3 blocks of 2x3x2
    # Each block: 7 (Strassen) + 4 (Tail) = 11
    # Total 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A is 2x3, B is 3x6
    # B blocks: cols 0-1, 2-3, 4-5
    # For each block j in 0..2:
    #   A is 2x3, B_j is 3x2
    #   Strassen on A[:, 0:2] and B_j[0:2, :]
    #   Tail: A[:, 2] and B_j[2, :]
    
    col = 0
    for j in range(3):
        # Strassen part (7 terms)
        # A' = A[:, 0:2], B' = B_j[0:2, :]
        # M1 = (A00+A11)(B00+B10)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U indices: i*m + k. i in 0..1, k in 0..2
        # V indices: j*p + l. j in 0..2, l in 0..5
        # W indices: l*n + i. l in 0..5, i in 0..1
        
        # Block j uses B cols 2j, 2j+1
        c0 = 2 * j
        c1 = 2 * j + 1
        
        # Term 1: (A00+A11)(B00+B10)
        # U: A00 (0,0), A11 (1,1)
        # V: B00 (0, c0), B10 (1, c0)
        # W: Identity? No, Strassen combines.
        # C00 += M1 - M4 + M5 - M7
        # C01 += M3 + M4 + M7
        # C10 += M2 + M5 - M1 - M6
        # C11 += M1 - M2 + M3 + M6
        
        # We add 7 terms to the global rank.
        
        # T1
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*6+c0][col] = 1
        V[1*6+c0][col] = 1
        W[c0*2+0][col] = 1
        W[c0*2+1][col] = 1 # Wait, W is for output C.
        # Strassen output is 2x2.
        # C00 gets +1, C11 gets +1?
        # M1 contributes to C00 (+1), C11 (+1).
        # C00 is W[c0*2+0], C11 is W[c0*2+1]?
        # W index is k*n+i. k is col index of C (0..5), i is row (0..1).
        # C00: k=c0, i=0 -> W[c0*2+0]
        # C11: k=c1, i=1 -> W[c1*2+1]
        # So W[c0*2+0] += 1, W[c1*2+1] += 1.
        W[c0*2+0][col] = 1
        W[c1*2+1][col] = 1
        col += 1
        
        # T2: (A10+A11)B00
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*6+c0][col] = 1
        W[c0*2+1][col] = 1 # C10
        W[c1*2+1][col] = -1 # C11
        col += 1
        
        # T3: A00(B01-B11)
        U[0*3+0][col] = 1
        V[0*6+c1][col] = 1
        V[1*6+c1][col] = -1
        W[c0*2+0][col] = 1 # C00? No, C01
        # M3 contributes to C01 (+1), C11 (+1)
        W[c1*2+0][col] = 1
        W[c1*2+1][col] = 1
        col += 1
        
        # T4: A11(B10-B01)
        U[1*3+1][col] = 1
        V[1*6+c0][col] = 1
        V[0*6+c1][col] = -1
        W[c0*2+0][col] = -1 # C00
        W[c1*2+0][col] = 1 # C01
        col += 1
        
        # T5: (A00+A01)B11
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*6+c1][col] = 1
        W[c0*2+0][col] = 1 # C00
        W[c0*2+1][col] = 1 # C10
        col += 1
        
        # T6: (A10-A00)(B00+B01)
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*6+c0][col] = 1
        V[0*6+c1][col] = 1
        W[c0*2+1][col] = 1 # C10
        W[c1*2+1][col] = 1 # C11
        col += 1
        
        # T7: (A01-A11)(B10+B11)
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*6+c0][col] = 1
        V[1*6+c1][col] = 1
        W[c0*2+0][col] = -1 # C00
        W[c1*2+0][col] = -1 # C01
        col += 1
        
        # Tail: c * B2 (4 terms)
        # A[:, 2] is c, f. B_j[2, :] is k, l.
        # C00 += c*k
        # C01 += c*l
        # C10 += f*k
        # C11 += f*l
        
        # T8: c*k
        U[0*3+2][col] = 1
        V[2*6+c0][col] = 1
        W[c0*2+0][col] = 1
        col += 1
        
        # T9: c*l
        U[0*3+2][col] = 1
        V[2*6+c1][col] = 1
        W[c1*2+0][col] = 1
        col += 1
        
        # T10: f*k
        U[1*3+2][col] = 1
        V[2*6+c0][col] = 1
        W[c0*2+1][col] = 1
        col += 1
        
        # T11: f*l
        U[1*3+2][col] = 1
        V[2*6+c1][col] = 1
        W[c1*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
