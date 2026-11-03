DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    # We'll build columns incrementally.
    U_cols = []  # each is a dict (i,j)->coeff
    V_cols = []
    W_cols = []
    
    def add_col(a, b, c):
        # a: dict (i,j)->coeff, b: dict (j,k)->coeff, c: dict (i,k)->coeff
        U_cols.append(a)
        V_cols.append(b)
        W_cols.append(c)
    
    # Strassen blocks
    i_block = (0, 1)  # i1, i2
    j_blocks = [(0,1), (2,3)]
    k_blocks = [(0,1), (2,3), (4,5)]
    
    for (i1, i2) in [i_block]:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                # 7 Strassen products
                # a forms
                a1 = {(i1,j1):1, (i2,j2):1}
                a2 = {(i2,j1):1, (i2,j2):1}
                a3 = {(i1,j1):1}
                a4 = {(i2,j2):1}
                a5 = {(i1,j1):1, (i1,j2):1}
                a6 = {(i2,j1):1, (i1,j1):-1}
                a7 = {(i1,j2):1, (i2,j2):-1}
                # b forms
                b1 = {(j1,k1):1, (j2,k2):1}
                b2 = {(j1,k1):1}
                b3 = {(j1,k2):1, (j2,k2):-1}
                b4 = {(j2,k1):1, (j1,k1):-1}
                b5 = {(j2,k2):1}
                b6 = {(j1,k1):1, (j1,k2):1}
                b7 = {(j1,k2):1, (j2,k2):1}
                # c forms
                c1 = {(i1,k1):1, (i2,k2):1}
                c2 = {(i2,k1):1, (i2,k2):-1}
                c3 = {(i1,k2):1, (i2,k2):1}
                c4 = {(i1,k1):1, (i2,k1):1}
                c5 = {(i1,k1):-1, (i1,k2):1}
                c6 = {(i2,k2):1}
                c7 = {(i1,k1):1}
                
                add_col(a1, b1, c1)
                add_col(a2, b2, c2)
                add_col(a3, b3, c3)
                add_col(a4, b4, c4)
                add_col(a5, b5, c5)
                add_col(a6, b6, c6)
                add_col(a7, b7, c7)
    
    # Schoolbook leftover
    covered = set()
    for (i1,i2) in [i_block]:
        for (j1,j2) in j_blocks:
            for (k1,k2) in k_blocks:
                for i in (i1,i2):
                    for j in (j1,j2):
                        for k in (k1,k2):
                            covered.add((i,j,k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    add_col({(i,j):1}, {(j,k):1}, {(i,k):1})
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for col in range(rank):
        for (i,j), coeff in U_cols[col].items():
            U[i*m + j][col] = coeff
        for (j,k), coeff in V_cols[col].items():
            V[j*p + k][col] = coeff
        for (i,k), coeff in W_cols[col].items():
            W[k*n + i][col] = coeff
    return {"U": U, "V": V, "W": W}
