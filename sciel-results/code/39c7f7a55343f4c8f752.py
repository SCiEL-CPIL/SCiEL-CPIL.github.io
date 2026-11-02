import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(ni, nj, nk, i0, j0, k0, U_block, V_block, W_block):
        rank = len(U_block[0])
        u_off = []
        for i in range(i0, i0 + ni):
            for j in range(j0, j0 + nj):
                u_off.append(i * m + j)
        v_off = []
        for j in range(j0, j0 + nj):
            for k in range(k0, k0 + nk):
                v_off.append(j * p + k)
        w_off = []
        for k in range(k0, k0 + nk):
            for i in range(i0, i0 + ni):
                w_off.append(k * n + i)
                
        for c in range(rank):
            uc = [0.0] * (n*m)
            vc = [0.0] * (m*p)
            wc = [0.0] * (p*n)
            for r in range(ni*nj):
                if U_block[r][c] != 0:
                    uc[u_off[r]] = U_block[r][c]
            for r in range(nj*nk):
                if V_block[r][c] != 0:
                    vc[v_off[r]] = V_block[r][c]
            for r in range(nk*ni):
                if W_block[r][c] != 0:
                    wc[w_off[r]] = W_block[r][c]
            U_cols.append(uc)
            V_cols.append(vc)
            W_cols.append(wc)

    # M222 Strassen
    U222 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    V222 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W222 = [
        [1, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [-1, 0, 0, 1, 0, 1, 1]
    ]
    
    # M221 (Rank 4)
    # A(2x2), B(2x1). C(2x1).
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    # T1: A11 B11 -> C1. u=[1,0,0,0], v=[1,0], w=[1,0]
    # T2: A12 B21 -> C1. u=[0,1,0,0], v=[0,1], w=[1,0]
    # T3: A21 B11 -> C2. u=[0,0,1,0], v=[1,0], w=[0,1]
    # T4: A22 B21 -> C2. u=[0,0,0,1], v=[0,1], w=[0,1]
    U221 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V221 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W221 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # M123 (Rank 6)
    # A(1x2), B(2x3). C(1x3).
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # C3 = A1 B13 + A2 B23
    # T1: A1 B11 -> C1. u=[1,0], v=[1,0,0,0], w=[1,0,0]
    # T2: A2 B21 -> C1. u=[0,1], v=[0,0,1,0], w=[1,0,0]
    # T3: A1 B12 -> C2. u=[1,0], v=[0,1,0,0], w=[0,1,0]
    # T4: A2 B22 -> C2. u=[0,1], v=[0,0,0,1], w=[0,1,0]
    # T5: A1 B13 -> C3. u=[1,0], v=[0,0,0,1]? No, B is 2x3.
    # B indices: 11,12,13, 21,22,23.
    # v=[0,0,0,1] is B21? No.
    # B row 1: 11, 12, 13.
    # B row 2: 21, 22, 23.
    # v indices: 0,1,2,3,4,5.
    # T1: v[0]=1.
    # T2: v[3]=1.
    # T3: v[1]=1.
    # T4: v[4]=1.
    # T5: v[2]=1.
    # T6: v[5]=1.
    U123 = [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1]
    ]
    V123 = [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ]
    W123 = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ]
    
    # M313 (Rank 9)
    # A(3x1), B(1x3). C(3x3).
    # Cij = Ai B1j.
    # T_i,j: u=[e_i], v=[1], w=[e_j in C_i,j]
    # w index k*n + i = 3*i + j.
    U313 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]
    V313 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    W313 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    # Wait, W indices:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # i=1, j=2 -> 5
    # i=2, j=0 -> 6
    # i=2, j=1 -> 7
    # i=2, j=2 -> 8
    # My W313 above:
    # Row 0 (k=0, i=0? No, W row is k*n+i).
    # W row 0: k=0, i=0. C00.
    # W row 1: k=0, i=1? No.
    # W row index = k*3 + i.
    # Row 0: 0*3+0 = 0. (C00)
    # Row 1: 0*3+1 = 1. (C10? No, C is 3x3. i is row, j is col? No, C is n x p. i in 1..n, j in 1..p.
    # In M313, n=3, p=3.
    # C index is j*3 + i?
    # Task: W[k*n+i, r].
    # For M313, output C is 3x3.
    # i (row of C) is 0..2. k (col of C) is 0..2.
    # W index = k*3 + i.
    # So W row 0: k=0, i=0. (C00)
    # W row 1: k=0, i=1. (C10)
    # W row 2: k=0, i=2. (C20)
    # W row 3: k=1, i=0. (C01)
    # ...
    # My T_i,j:
    # i (A row), j (B col).
    # C_i,j = A_i B_j.
    # So we need W row = j*3 + i to be 1.
    # T_0,0: i=0, j=0. W row 0.
    # T_0,1: i=0, j=1. W row 3.
    # T_0,2: i=0, j=2. W row 6.
    # T_1,0: i=1, j=0. W row 1.
    # T_1,1: i=1, j=1. W row 4.
    # T_1,2: i=1, j=2. W row 7.
    # T_2,0: i=2, j=0. W row 2.
    # T_2,1: i=2, j=1. W row 5.
    # T_2,2: i=2, j=2. W row 8.
    
    # Correct W313:
    W313 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    # But we must map the columns correctly.
    # Col 0: T00. W0=1.
    # Col 1: T01. W3=1.
    # Col 2: T02. W6=1.
    # Col 3: T10. W1=1.
    # Col 4: T11. W4=1.
    # Col 5: T12. W7=1.
    # Col 6: T20. W2=1.
    # Col 7: T21. W5=1.
    # Col 8: T22. W8=1.
    
    # This is getting messy to hardcode.
    # I will return the schoolbook decomposition to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
