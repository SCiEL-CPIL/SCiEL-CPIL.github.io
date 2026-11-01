import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Outer product of A[:, 2] and B[2, :]
    # A is 2x3, B is 3x9
    # A[:, 2] is 2x1 vector. Indices in U: 2, 5
    # B[2, :] is 1x9 vector. Indices in V: 18..26
    # C is 2x9. Indices in W: k*2 + i
    for i in range(2):
        for k in range(9):
            u_idx = i * 3 + 2
            v_idx = 2 * 9 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: Matrix multiplication of A[:, :2] (2x2) and B[:2, :] (2x9)
    # Split B[:2, :] into 3 blocks of 2x3
    for b in range(3):
        c0 = 3 * b
        c1 = 3 * b + 1
        c2 = 3 * b + 2
        
        # Strassen for 2x2 x 2 (cols c0, c1)
        # A indices: 0 (0,0), 1 (0,1), 3 (1,0), 4 (1,1)
        # B indices: 
        # e = B[0, c0] -> 0*9 + c0
        # f = B[0, c1] -> 0*9 + c1
        # g = B[1, c0] -> 1*9 + c0
        # h = B[1, c1] -> 1*9 + c1
        # C indices:
        # C00 = C[0, c0] -> c0*2 + 0
        # C01 = C[0, c1] -> c1*2 + 0
        # C10 = C[1, c0] -> c0*2 + 1
        # C11 = C[1, c1] -> c1*2 + 1
        
        # P1 = (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[0*9+c0][col] = 1; V[1*9+c1][col] = 1
        W[c0*2+0][col] = 1; W[c1*2+1][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[3][col] = 1; U[4][col] = 1
        V[0*9+c0][col] = 1
        W[c0*2+1][col] = 1
        col += 1
        
        # P3 = a(f-h)
        U[0][col] = 1
        V[0*9+c1][col] = 1; V[1*9+c1][col] = -1
        W[c1*2+0][col] = 1; W[c1*2+1][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[4][col] = 1
        V[0*9+c0][col] = -1; V[1*9+c0][col] = 1
        W[c0*2+0][col] = 1; W[c0*2+1][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[0][col] = 1; U[1][col] = 1
        V[1*9+c1][col] = 1
        W[c0*2+0][col] = -1; W[c1*2+0][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[0][col] = -1; U[3][col] = 1
        V[0*9+c0][col] = 1; V[0*9+c1][col] = 1
        W[c1*2+1][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[1][col] = 1; U[4][col] = -1
        V[1*9+c0][col] = 1; V[1*9+c1][col] = 1
        W[c0*2+0][col] = 1
        col += 1
        
        # Schoolbook for 2x2 x 1 (col c2)
        # e2 = B[0, c2], g2 = B[1, c2]
        # C02 = C[0, c2], C12 = C[1, c2]
        
        # a * e2 -> C02
        U[0][col] = 1
        V[0*9+c2][col] = 1
        W[c2*2+0][col] = 1
        col += 1
        
        # b * g2 -> C02
        U[1][col] = 1
        V[1*9+c2][col] = 1
        W[c2*2+0][col] = 1
        col += 1
        
        # c * e2 -> C12
        U[3][col] = 1
        V[0*9+c2][col] = 1
        W[c2*2+1][col] = 1
        col += 1
        
        # d * g2 -> C12
        U[4][col] = 1
        V[1*9+c2][col] = 1
        W[c2*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
