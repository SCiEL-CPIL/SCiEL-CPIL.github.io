DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    terms = []
    
    def strassen_2x2_2x2(i_off, j_off, k_off):
        s = []
        s.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        ))
        s.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(0,0):1, (1,1):-1}
        ))
        s.append((
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(0,1):1, (1,1):1}
        ))
        s.append((
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(0,0):1, (1,0):1}
        ))
        s.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,1):-1, (1,0):-1}
        ))
        s.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ))
        s.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1, (1,1):-1}
        ))
        return s
    
    def schoolbook_2x2_2x1(i_off, j_off, k_off):
        s = []
        for i in range(2):
            for j in range(2):
                s.append((
                    {(i,j):1},
                    {(j,0):1},
                    {(0,i):1}
                ))
        return s
    
    # Block 1: C[:, 0:2]
    for term in strassen_2x2_2x2(0, 0, 0):
        terms.append(term)
    for term in strassen_2x2_2x2(0, 2, 0):
        terms.append(term)
    
    # Block 2: C[:, 2:4]
    for term in strassen_2x2_2x2(0, 0, 2):
        terms.append(term)
    for term in strassen_2x2_2x2(0, 2, 2):
        terms.append(term)
    
    # Block 3: C[:, 4]
    for term in schoolbook_2x2_2x1(0, 0, 4):
        terms.append(term)
    for term in schoolbook_2x2_2x1(0, 2, 4):
        terms.append(term)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for (i, j), coeff in u.items():
            U[i*m + j][c] += coeff
        for (j, k), coeff in v.items():
            V[j*p + k][c] += coeff
        for (k, i), coeff in w.items():
            W[k*n + i][c] += coeff
    
    return {"U": U, "V": V, "W": W}
