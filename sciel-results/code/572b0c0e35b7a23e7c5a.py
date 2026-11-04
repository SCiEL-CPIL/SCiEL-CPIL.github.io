DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def alloc():
        nonlocal col
        c = col
        col += 1
        return c

    def add_schoolbook(i, j, k):
        c = alloc()
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1

    def add_strassen_222(i_base, j_base, k_base):
        # M1 = (A00+A11)(B00+B11)
        c = alloc()
        U[i_base * m + j_base][c] = 1
        U[(i_base+1) * m + (j_base+1)][c] = 1
        V[j_base * p + k_base][c] = 1
        V[(j_base+1) * p + (k_base+1)][c] = 1
        W[k_base * n + i_base][c] = 1
        W[(k_base+1) * n + (i_base+1)][c] = 1
        
        # M2 = (A10+A11)B00
        c = alloc()
        U[(i_base+1) * m + j_base][c] = 1
        U[(i_base+1) * m + (j_base+1)][c] = 1
        V[j_base * p + k_base][c] = 1
        W[k_base * n + (i_base+1)][c] = 1
        W[(k_base+1) * n + (i_base+1)][c] = -1
        
        # M3 = A00(B10-B11)
        c = alloc()
        U[i_base * m + j_base][c] = 1
        V[(j_base+1) * p + k_base][c] = 1
        V[(j_base+1) * p + (k_base+1)][c] = -1
        W[k_base * n + i_base][c] = 1
        W[(k_base+1) * n + i_base][c] = 1
        
        # M4 = A11(B01-B00)
        c = alloc()
        U[(i_base+1) * m + (j_base+1)][c] = 1
        V[j_base * p + (k_base+1)][c] = 1
        V[j_base * p + k_base][c] = -1
        W[k_base * n + i_base][c] = 1
        W[k_base * n + (i_base+1)][c] = 1
        
        # M5 = (A00+A01)B11
        c = alloc()
        U[i_base * m + j_base][c] = 1
        U[i_base * m + (j_base+1)][c] = 1
        V[(j_base+1) * p + (k_base+1)][c] = 1
        W[k_base * n + i_base][c] = -1
        W[(k_base+1) * n + i_base][c] = 1
        
        # M6 = (A10-A00)(B00+B01)
        c = alloc()
        U[(i_base+1) * m + j_base][c] = 1
        U[i_base * m + j_base][c] = -1
        V[j_base * p + k_base][c] = 1
        V[j_base * p + (k_base+1)][c] = 1
        W[(k_base+1) * n + (i_base+1)][c] = 1
        
        # M7 = (A01-A11)(B10+B11)
        c = alloc()
        U[i_base * m + (j_base+1)][c] = 1
        U[(i_base+1) * m + (j_base+1)][c] = -1
        V[(j_base+1) * p + k_base][c] = 1
        V[(j_base+1) * p + (k_base+1)][c] = 1
        W[k_base * n + i_base][c] = 1
        W[(k_base+1) * n + (i_base+1)][c] = 1

    # Apply Strassen to 4 blocks for i in {0,1}
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 2, 2)
    
    # Schoolbook for remaining i in {0,1}
    # j in {0,1}, k=4
    for j in [0, 1]:
        add_schoolbook(0, j, 4)
        add_schoolbook(1, j, 4)
    # j in {2,3}, k=4
    for j in [2, 3]:
        add_schoolbook(0, j, 4)
        add_schoolbook(1, j, 4)
    # j=4, k in {0,1}
    for k in [0, 1]:
        add_schoolbook(0, 4, k)
        add_schoolbook(1, 4, k)
    # j=4, k in {2,3}
    for k in [2, 3]:
        add_schoolbook(0, 4, k)
        add_schoolbook(1, 4, k)
    # j=4, k=4
    add_schoolbook(0, 4, 4)
    add_schoolbook(1, 4, 4)
    
    # Schoolbook for i=2
    for j in range(5):
        for k in range(5):
            add_schoolbook(2, j, k)
            
    return {"U": U, "V": V, "W": W}
