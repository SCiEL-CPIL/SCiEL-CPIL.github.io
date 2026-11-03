"""Matrix multiplication 4x4x7 tensor decomposition."""
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    # 4 A-blocks x 3 B-col-pairs (Strassen 2x2x2, rank 7 each) + 4 A-blocks x 1 B-col (schoolbook 2x2x1, rank 4 each)
    R = 4 * 3 * 7 + 4 * 4  # = 84 + 16 = 100
    
    U = [[0] * R for _ in range(n * m)]  # (16, 100)
    V = [[0] * R for _ in range(m * p)]  # (28, 100)
    W = [[0] * R for _ in range(p * n)]  # (28, 100)
    
    def add_strassen_2x2x2(U, V, W, ar, ac, bc, start_col):
        # A block at rows ar,ar+1; cols ac,ac+1
        # B block at rows ar,ar+1; cols bc,bc+1
        # C block at rows ar,ar+1; cols bc,bc+1
        a = ar * m + ac
        b = ar * m + ac + 1
        c = (ar + 1) * m + ac
        d = (ar + 1) * m + ac + 1
        
        e = ar * p + bc
        f = ar * p + bc + 1
        g = (ar + 1) * p + bc
        h = (ar + 1) * p + bc + 1
        
        C11 = bc * n + ar
        C12 = (bc + 1) * n + ar
        C21 = bc * n + (ar + 1)
        C22 = (bc + 1) * n + (ar + 1)
        
        col = start_col
        # M1 = (a+d)(e+h) -> C11, C22
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[C11][col] += 1; W[C22][col] += 1
        col += 1
        # M2 = (c+d)e -> C21, -C22
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[C21][col] += 1; W[C22][col] -= 1
        col += 1
        # M3 = a(f-h) -> C12, C22
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[C12][col] += 1; W[C22][col] += 1
        col += 1
        # M4 = d(g-e) -> C11, C21
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[C11][col] += 1; W[C21][col] += 1
        col += 1
        # M5 = (a+b)h -> -C11, C12
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[C11][col] -= 1; W[C12][col] += 1
        col += 1
        # M6 = (c-a)(e+f) -> C22
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[C22][col] += 1
        col += 1
        # M7 = (b-d)(g+h) -> C11
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[C11][col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2x1(U, V, W, ar, ac, bc, start_col):
        # A block at rows ar,ar+1; cols ac,ac+1
        # B at rows ar,ar+1; col bc
        # C at rows ar,ar+1; col bc
        a = ar * m + ac
        b = ar * m + ac + 1
        c = (ar + 1) * m + ac
        d = (ar + 1) * m + ac + 1
        
        B0 = ar * p + bc
        B1 = (ar + 1) * p + bc
        
        C11 = bc * n + ar
        C21 = bc * n + (ar + 1)
        
        col = start_col
        # a * B0 -> C11
        U[a][col] = 1; V[B0][col] = 1; W[C11][col] = 1
        col += 1
        # b * B1 -> C11
        U[b][col] = 1; V[B1][col] = 1; W[C11][col] = 1
        col += 1
        # c * B0 -> C21
        U[c][col] = 1; V[B0][col] = 1; W[C21][col] = 1
        col += 1
        # d * B1 -> C21
        U[d][col] = 1; V[B1][col] = 1; W[C21][col] = 1
        col += 1
        
        return col
    
    col = 0
    for ar in [0, 2]:       # A row block
        for ac in [0, 2]:   # A col block
            for bc in [0, 2, 4]:  # B col pairs
                col = add_strassen_2x2x2(U, V, W, ar, ac, bc, col)
            # B col 6
            col = add_schoolbook_2x2x1(U, V, W, ar, ac, 6, col)
    
    assert col == R, f"Column mismatch: col={col}, R={R}"
    
    return {"U": U, "V": V, "W": W}
