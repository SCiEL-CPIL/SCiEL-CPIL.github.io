import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Start with empty lists for columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """
        u: vector of length n*m (flattened A)
        v: vector of length m*p (flattened B)
        w: vector of length p*n (flattened C^T)
        """
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def idx_U(i, j):
        return i * m + j
        
    def idx_V(j, k):
        return j * p + k
        
    def idx_W(k, i):
        return k * n + i

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A_TL is 2x2, B_TL is 2x2
    # Indices in A: (0,0), (0,1), (1,0), (1,1)
    # Indices in B: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00 + A11) * (B00 + B11)
    u = [0]*(n*m); u[idx_U(0,0)] = 1; u[idx_U(1,1)] = 1
    v = [0]*(m*p); v[idx_V(0,0)] = 1; v[idx_V(1,1)] = 1
    # Contribution to C00, C01, C10, C11
    # C00 gets +1, C01 gets +1, C10 gets +1, C11 gets +1
    # Wait, M1 contributes to C00, C01, C10, C11 with specific signs in the final reconstruction.
    # Let's use the standard Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    w = [0]*(p*n)
    w[idx_W(0,0)] = 1 # C00
    w[idx_W(1,0)] = 1 # C01
    w[idx_W(0,1)] = 1 # C10
    w[idx_W(1,1)] = 1 # C11
    add_term(u, v, w)
    
    # M2 = (A10 + A11) * B00
    u = [0]*(n*m); u[idx_U(1,0)] = 1; u[idx_U(1,1)] = 1
    v = [0]*(m*p); v[idx_V(0,0)] = 1
    w = [0]*(p*n)
    w[idx_W(0,1)] = 1 # C10
    w[idx_W(1,1)] = -1 # C11
    add_term(u, v, w)
    
    # M3 = A00 * (B01 - B11)
    u = [0]*(n*m); u[idx_U(0,0)] = 1
    v = [0]*(m*p); v[idx_V(0,1)] = 1; v[idx_V(1,1)] = -1
    w = [0]*(p*n)
    w[idx_W(1,0)] = 1 # C01
    w[idx_W(1,1)] = 1 # C11
    add_term(u, v, w)
    
    # M4 = A11 * (B10 - B00)
    u = [0]*(n*m); u[idx_U(1,1)] = 1
    v = [0]*(m*p); v[idx_V(1,0)] = 1; v[idx_V(0,0)] = -1
    w = [0]*(p*n)
    w[idx_W(0,0)] = 1 # C00
    w[idx_W(0,1)] = 1 # C10
    add_term(u, v, w)
    
    # M5 = (A00 + A01) * B11
    u = [0]*(n*m); u[idx_U(0,0)] = 1; u[idx_U(0,1)] = 1
    v = [0]*(m*p); v[idx_V(1,1)] = 1
    w = [0]*(p*n)
    w[idx_W(0,0)] = -1 # C00
    w[idx_W(1,0)] = 1 # C01
    add_term(u, v, w)
    
    # M6 = (A10 - A00) * (B00 + B01)
    u = [0]*(n*m); u[idx_U(1,0)] = 1; u[idx_U(0,0)] = -1
    v = [0]*(m*p); v[idx_V(0,0)] = 1; v[idx_V(0,1)] = 1
    w = [0]*(p*n)
    w[idx_W(1,1)] = 1 # C11
    add_term(u, v, w)
    
    # M7 = (A01 - A11) * (B10 + B11)
    u = [0]*(n*m); u[idx_U(0,1)] = 1; u[idx_U(1,1)] = -1
    v = [0]*(m*p); v[idx_V(1,0)] = 1; v[idx_V(1,1)] = 1
    w = [0]*(p*n)
    w[idx_W(0,0)] = 1 # C00
    w[idx_W(0,1)] = 1 # C10
    add_term(u, v, w)
    
    # 2. Schoolbook for remaining terms
    # A_TR (2x1) * B_BL (1x2) -> C_TL (2x2)
    # A_TR is A[0:2, 2], B_BL is B[2, 0:2]
    # Terms:
    # A02 * B20 -> C00
    u = [0]*(n*m); u[idx_U(0,2)] = 1
    v = [0]*(m*p); v[idx_V(2,0)] = 1
    w = [0]*(p*n); w[idx_W(0,0)] = 1
    add_term(u, v, w)
    
    # A02 * B21 -> C01
    u = [0]*(n*m); u[idx_U(0,2)] = 1
    v = [0]*(m*p); v[idx_V(2,1)] = 1
    w = [0]*(p*n); w[idx_W(1,0)] = 1
    add_term(u, v, w)
    
    # A12 * B20 -> C10
    u = [0]*(n*m); u[idx_U(1,2)] = 1
    v = [0]*(m*p); v[idx_V(2,0)] = 1
    w = [0]*(p*n); w[idx_W(0,1)] = 1
    add_term(u, v, w)
    
    # A12 * B21 -> C11
    u = [0]*(n*m); u[idx_U(1,2)] = 1
    v = [0]*(m*p); v[idx_V(2,1)] = 1
    w = [0]*(p*n); w[idx_W(1,1)] = 1
    add_term(u, v, w)
    
    # A_TL (2x2) * B_TR (2x5) -> C_TR (2x5)
    # B_TR is B[0:2, 2:7]
    # A00 * B0k -> C0k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(0,0)] = 1
        v = [0]*(m*p); v[idx_V(0,k)] = 1
        w = [0]*(p*n); w[idx_W(k,0)] = 1
        add_term(u, v, w)
        
    # A01 * B1k -> C0k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(0,1)] = 1
        v = [0]*(m*p); v[idx_V(1,k)] = 1
        w = [0]*(p*n); w[idx_W(k,0)] = 1
        add_term(u, v, w)
        
    # A10 * B0k -> C1k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(1,0)] = 1
        v = [0]*(m*p); v[idx_V(0,k)] = 1
        w = [0]*(p*n); w[idx_W(k,1)] = 1
        add_term(u, v, w)
        
    # A11 * B1k -> C1k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(1,1)] = 1
        v = [0]*(m*p); v[idx_V(1,k)] = 1
        w = [0]*(p*n); w[idx_W(k,1)] = 1
        add_term(u, v, w)
        
    # A_TR (2x1) * B_BR (1x5) -> C_TR (2x5)
    # B_BR is B[2, 2:7]
    # A02 * B2k -> C0k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(0,2)] = 1
        v = [0]*(m*p); v[idx_V(2,k)] = 1
        w = [0]*(p*n); w[idx_W(k,0)] = 1
        add_term(u, v, w)
        
    # A12 * B2k -> C1k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(1,2)] = 1
        v = [0]*(m*p); v[idx_V(2,k)] = 1
        w = [0]*(p*n); w[idx_W(k,1)] = 1
        add_term(u, v, w)
        
    # A_BL (1x2) * B_TL (2x2) -> C_BL (1x2)
    # A20 * B00 -> C20
    u = [0]*(n*m); u[idx_U(2,0)] = 1
    v = [0]*(m*p); v[idx_V(0,0)] = 1
    w = [0]*(p*n); w[idx_W(0,2)] = 1
    add_term(u, v, w)
    
    # A20 * B01 -> C21
    u = [0]*(n*m); u[idx_U(2,0)] = 1
    v = [0]*(m*p); v[idx_V(0,1)] = 1
    w = [0]*(p*n); w[idx_W(1,2)] = 1
    add_term(u, v, w)
    
    # A21 * B10 -> C20
    u = [0]*(n*m); u[idx_U(2,1)] = 1
    v = [0]*(m*p); v[idx_V(1,0)] = 1
    w = [0]*(p*n); w[idx_W(0,2)] = 1
    add_term(u, v, w)
    
    # A21 * B11 -> C21
    u = [0]*(n*m); u[idx_U(2,1)] = 1
    v = [0]*(m*p); v[idx_V(1,1)] = 1
    w = [0]*(p*n); w[idx_W(1,2)] = 1
    add_term(u, v, w)
    
    # A_BR (1x1) * B_BL (1x2) -> C_BL (1x2)
    # A22 * B20 -> C20
    u = [0]*(n*m); u[idx_U(2,2)] = 1
    v = [0]*(m*p); v[idx_V(2,0)] = 1
    w = [0]*(p*n); w[idx_W(0,2)] = 1
    add_term(u, v, w)
    
    # A22 * B21 -> C21
    u = [0]*(n*m); u[idx_U(2,2)] = 1
    v = [0]*(m*p); v[idx_V(2,1)] = 1
    w = [0]*(p*n); w[idx_W(1,2)] = 1
    add_term(u, v, w)
    
    # A_BL (1x2) * B_TR (2x5) -> C_BR (1x5)
    # A20 * B0k -> C2k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(2,0)] = 1
        v = [0]*(m*p); v[idx_V(0,k)] = 1
        w = [0]*(p*n); w[idx_W(k,2)] = 1
        add_term(u, v, w)
        
    # A21 * B1k -> C2k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(2,1)] = 1
        v = [0]*(m*p); v[idx_V(1,k)] = 1
        w = [0]*(p*n); w[idx_W(k,2)] = 1
        add_term(u, v, w)
        
    # A_BR (1x1) * B_BR (1x5) -> C_BR (1x5)
    # A22 * B2k -> C2k
    for k in range(2, 7):
        u = [0]*(n*m); u[idx_U(2,2)] = 1
        v = [0]*(m*p); v[idx_V(2,k)] = 1
        w = [0]*(p*n); w[idx_W(k,2)] = 1
        add_term(u, v, w)
        
    # Convert to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
