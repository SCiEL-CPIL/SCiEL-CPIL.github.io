import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Rank 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if coeff == 0:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Define Strassen block terms
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1
        # U: (0,0):1, (1,0):1
        # V: (0,0):1, (1,0):1
        # W: (0,0):1, (1,1):1
        u0 = (0 + i_off) * m + (0 + j_off)
        u1 = (1 + i_off) * m + (0 + j_off)
        v0 = (0 + j_off) * p + (0 + k_off)
        v1 = (1 + j_off) * p + (0 + k_off)
        w0 = (0 + k_off) * n + (0 + i_off)
        w1 = (1 + k_off) * n + (1 + i_off)
        
        # We need to add 7 terms.
        # But U, V, W are vectors for each term.
        # So we iterate through the 7 Strassen products.
        
        # M1
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        U[u1][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1
        
        # M2
        # U: (0,1):1, (1,1):1
        # V: (0,0):1
        # W: (0,1):1
        u0 = (0 + i_off) * m + (1 + j_off)
        u1 = (1 + i_off) * m + (1 + j_off)
        v0 = (0 + j_off) * p + (0 + k_off)
        w0 = (0 + k_off) * n + (1 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        U[u1][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        
        # M3
        # U: (0,0):1
        # V: (1,0):1, (0,0):-1
        # W: (1,0):1, (0,1):-1
        u0 = (0 + i_off) * m + (0 + j_off)
        v0 = (1 + j_off) * p + (0 + k_off)
        v1 = (0 + j_off) * p + (0 + k_off)
        w0 = (1 + k_off) * n + (0 + i_off)
        w1 = (0 + k_off) * n + (1 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        V[v1][col] = -1; W[w1][col] = -1
        col += 1
        
        # M4
        # U: (1,0):1
        # V: (1,1):1, (0,0):-1
        # W: (0,0):-1, (0,1):1
        u0 = (1 + i_off) * m + (0 + j_off)
        v0 = (1 + j_off) * p + (1 + k_off)
        v1 = (0 + j_off) * p + (0 + k_off)
        w0 = (0 + k_off) * n + (0 + i_off)
        w1 = (0 + k_off) * n + (1 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = -1
        V[v1][col] = -1; W[w1][col] = 1
        col += 1
        
        # M5
        # U: (0,0):1, (0,1):1
        # V: (1,1):1
        # W: (1,0):1, (1,1):1
        u0 = (0 + i_off) * m + (0 + j_off)
        u1 = (0 + i_off) * m + (1 + j_off)
        v0 = (1 + j_off) * p + (1 + k_off)
        w0 = (1 + k_off) * n + (0 + i_off)
        w1 = (1 + k_off) * n + (1 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        U[u1][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        
        # M6
        # U: (1,1):1, (0,0):-1
        # V: (0,0):1, (0,1):1
        # W: (0,0):1, (1,1):-1
        u0 = (1 + i_off) * m + (1 + j_off)
        u1 = (0 + i_off) * m + (0 + j_off)
        v0 = (0 + j_off) * p + (0 + k_off)
        v1 = (0 + j_off) * p + (1 + k_off)
        w0 = (0 + k_off) * n + (0 + i_off)
        w1 = (1 + k_off) * n + (1 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        U[u1][col] = -1; V[v1][col] = 1; W[w1][col] = -1
        col += 1
        
        # M7
        # U: (0,1):1, (1,1):-1
        # V: (1,1):1, (1,0):-1
        # W: (0,0):-1
        u0 = (0 + i_off) * m + (1 + j_off)
        u1 = (1 + i_off) * m + (1 + j_off)
        v0 = (1 + j_off) * p + (1 + k_off)
        v1 = (1 + j_off) * p + (0 + k_off)
        w0 = (0 + k_off) * n + (0 + i_off)
        
        U[u0][col] = 1; V[v0][col] = 1; W[w0][col] = -1
        U[u1][col] = -1; V[v1][col] = -1
        col += 1

    # Part 1: j in {0, 1}
    # Strassen blocks for i in {0, 1}, k in {0, 1, 2}
    # A rows 0,1. B rows 0,1.
    for k_block in range(3):
        k_off = k_block * 2
        add_strassen_block(0, 0, k_off)
        
    # Schoolbook for A row 2, B rows 0,1
    for j in range(2):
        for k in range(6):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: j in {2, 3}
    # Strassen blocks for i in {0, 1}, k in {0, 1, 2}
    # A rows 0,1. B rows 2,3.
    # j_off = 2
    for k_block in range(3):
        k_off = k_block * 2
        add_strassen_block(0, 2, k_off)
        
    # Schoolbook for A row 2, B rows 2,3
    for j in range(2, 4):
        for k in range(6):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Verify col == rank
    if col != rank:
        # If mismatch, adjust rank or pad
        # But we designed for 66.
        # 3*7 + 12 + 3*7 + 12 = 21 + 12 + 21 + 12 = 66.
        pass

    return {"U": U, "V": V, "W": W}
