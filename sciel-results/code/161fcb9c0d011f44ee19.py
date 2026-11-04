"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 36 decomposition using Strassen for 2x2 blocks
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    def add_strassen(A_idx, B_idx, C_idx):
        A00, A01, A10, A11 = A_idx
        B00, B01, B10, B11 = B_idx
        C00, C01, C10, C11 = C_idx
        
        # M1
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A00] = 1; u[A11] = 1
        v[B00] = 1; v[B11] = 1
        w[C00] = 1; w[C11] = 1
        add_term(u, v, w)
        
        # M2
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A10] = 1; u[A11] = 1
        v[B00] = 1
        w[C10] = 1; w[C11] = -1
        add_term(u, v, w)
        
        # M3
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A00] = 1
        v[B01] = 1; v[B11] = -1
        w[C01] = 1; w[C11] = 1
        add_term(u, v, w)
        
        # M4
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A11] = 1
        v[B10] = 1; v[B00] = -1
        w[C00] = 1; w[C10] = 1
        add_term(u, v, w)
        
        # M5
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A00] = 1; u[A01] = 1
        v[B11] = 1
        w[C00] = -1; w[C01] = 1
        add_term(u, v, w)
        
        # M6
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A10] = 1; u[A00] = -1
        v[B00] = 1; v[B01] = 1
        w[C11] = 1
        add_term(u, v, w)
        
        # M7
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A01] = 1; u[A11] = -1
        v[B10] = 1; v[B11] = 1
        w[C00] = 1; w[C10] = 1
        add_term(u, v, w)

    def add_221(A_idx, B_idx, C_idx):
        A00, A01, A10, A11 = A_idx
        B0, B1 = B_idx
        C0, C1 = C_idx
        
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A00] = 1
        v[B0] = 1
        w[C0] = 1
        add_term(u, v, w)
        
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A01] = 1
        v[B1] = 1
        w[C0] = 1
        add_term(u, v, w)
        
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A10] = 1
        v[B0] = 1
        w[C1] = 1
        add_term(u, v, w)
        
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[A11] = 1
        v[B1] = 1
        w[C1] = 1
        add_term(u, v, w)

    A_idx_00 = [0, 1, 4, 5]
    A_idx_01 = [2, 3, 6, 7]
    
    B_idx_00 = [0, 1, 5, 6]
    B_idx_01 = [2, 3, 7, 8]
    B_idx_10 = [10, 11, 15, 16]
    B_idx_11 = [12, 13, 17, 18]
    
    B_idx_02 = [4, 9]
    B_idx_12 = [14, 19]
    
    C_idx_cols_01 = [0, 1, 2, 3]
    C_idx_cols_23 = [4, 5, 6, 7]
    C_idx_col_4 = [8, 9]
    
    add_strassen(A_idx_00, B_idx_00, C_idx_cols_01)
    add_strassen(A_idx_00, B_idx_01, C_idx_cols_23)
    add_strassen(A_idx_01, B_idx_10, C_idx_cols_01)
    add_strassen(A_idx_01, B_idx_11, C_idx_cols_23)
    
    add_221(A_idx_00, B_idx_02, C_idx_col_4)
    add_221(A_idx_01, B_idx_12, C_idx_col_4)
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n*m):
            if u[i] != 0:
                U[i][r] = u[i]
        for i in range(m*p):
            if v[i] != 0:
                V[i][r] = v[i]
        for i in range(p*n):
            if w[i] != 0:
                W[i][r] = w[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
