DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    
    def add_strassen(A_row_offset, A_col_offset, B_row_offset, B_col_offset):
        nonlocal col
        # A block is 2x2 at rows A_row_offset:A_row_offset+2, cols A_col_offset:A_col_offset+2
        # B block is 2x2 at rows B_row_offset:B_row_offset+2, cols B_col_offset:B_col_offset+2
        # C block is 2x2 at rows A_row_offset:A_row_offset+2, cols B_col_offset:B_col_offset+2
        
        # Strassen multiplications
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # A entries: a=A[ar,ac], b=A[ar,ac+1], c=A[ar+1,ac], d=A[ar+1,ac+1]
        # B entries: e=B[br,bc], f=B[br,bc+1], g=B[br+1,bc], h=B[br+1,bc+1]
        # C entries: C[ar,bc], C[ar,bc+1], C[ar+1,bc], C[ar+1,bc+1]
        
        ar, ac = A_row_offset, A_col_offset
        br, bc = B_row_offset, B_col_offset
        
        # m1 = (a+d)(e+h)
        U[ar*m + ac][col] += 1    # a
        U[(ar+1)*m + (ac+1)][col] += 1  # d
        V[br*p + bc][col] += 1    # e
        V[(br+1)*p + (bc+1)][col] += 1  # h
        # C contribution: C[ar,bc] += m1, C[ar+1,bc+1] += m1
        W[bc*n + ar][col] += 1
        W[(bc+1)*n + (ar+1)][col] += 1
        col += 1
        
        # m2 = (c+d)e
        U[(ar+1)*m + ac][col] += 1  # c
        U[(ar+1)*m + (ac+1)][col] += 1  # d
        V[br*p + bc][col] += 1    # e
        # C contribution: C[ar+1,bc] += m2, C[ar+1,bc+1] -= m2
        W[bc*n + (ar+1)][col] += 1
        W[(bc+1)*n + (ar+1)][col] -= 1
        col += 1
        
        # m3 = a(f-h)
        U[ar*m + ac][col] += 1    # a
        V[br*p + (bc+1)][col] += 1  # f
        V[(br+1)*p + bc][col] -= 1  # -g... wait, f-h means f and -h
        # f = B[br, bc+1], h = B[br+1, bc+1]
        V[br*p + (bc+1)][col] += 1
        V[(br+1)*p + (bc+1)][col] -= 1
        # C contribution: C[ar,bc+1] += m3, C[ar+1,bc+1] += m3
        W[(bc+1)*n + ar][col] += 1
        W[(bc+1)*n + (ar+1)][col] += 1
        col += 1
        
        # m4 = d(g-e)
        U[(ar+1)*m + (ac+1)][col] += 1  # d
        V[(br+1)*p + bc][col] += 1  # g
        V[br*p + bc][col] -= 1  # -e
        # C contribution: C[ar,bc] += m4, C[ar+1,bc] += m4
        W[bc*n + ar][col] += 1
        W[bc*n + (ar+1)][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[ar*m + ac][col] += 1    # a
        U[ar*m + (ac+1)][col] += 1  # b
        V[(br+1)*p + (bc+1)][col] += 1  # h
        # C contribution: C[ar,bc] -= m5, C[ar,bc+1] += m5
        W[bc*n + ar][col] -= 1
        W[(bc+1)*n + ar][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[(ar+1)*m + ac][col] += 1  # c
        U[ar*m + ac][col] -= 1    # -a
        V[br*p + bc][col] += 1    # e
        V[br*p + (bc+1)][col] += 1  # f
        # C contribution: C[ar+1,bc+1] += m6
        W[(bc+1)*n + (ar+1)][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[ar*m + (ac+1)][col] += 1  # b
        U[(ar+1)*m + (ac+1)][col] -= 1  # -d
        V[(br+1)*p + bc][col] += 1  # g
        V[(br+1)*p + (bc+1)][col] += 1  # h
        # C contribution: C[ar,bc] += m7
        W[bc*n + ar][col] += 1
        col += 1
    
    # A1 B1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-5
    # Split B into 3 blocks of 2 cols each
    add_strassen(0, 0, 0, 0)   # A1 * B1a (cols 0-1)
    add_strassen(0, 0, 0, 2)   # A1 * B1b (cols 2-3)
    add_strassen(0, 0, 0, 4)   # A1 * B1c (cols 4-5)
    
    # A2 B2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-5
    add_strassen(0, 2, 2, 0)   # A2 * B2a (cols 0-1)
    add_strassen(0, 2, 2, 2)   # A2 * B2b (cols 2-3)
    add_strassen(0, 2, 2, 4)   # A2 * B2c (cols 4-5)
    
    return {"U": U, "V": V, "W": W}
