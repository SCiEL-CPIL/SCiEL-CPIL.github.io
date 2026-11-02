import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the matrices U, V, W
    # U: (n*m, rank) -> (20, R)
    # V: (m*p, rank) -> (25, R)
    # W: (p*n, rank) -> (20, R)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2(a_i, a_j, b_j, b_k, w_offsets):
        # a_i, a_j: indices for A block (2x2)
        # b_j, b_k: indices for B block (2x2)
        # w_offsets: dict mapping (i, k) in block to coefficient in C
        # Actually, w_vec is for the whole C matrix.
        # We need to map local (i, k) to global (i, k).
        pass

    # Helper to create zero vectors
    def zero_u(): return [0.0] * (n * m)
    def zero_v(): return [0.0] * (m * p)
    def zero_w(): return [0.0] * (p * n)

    def set_u(u, i, j, val):
        u[i * m + j] = val
        
    def set_v(v, j, k, val):
        v[j * p + k] = val
        
    def set_w(w, k, i, val):
        w[k * n + i] = val

    # Strassen 2x2x2
    # A block rows i0, i1; cols j0, j1
    # B block rows j0, j1; cols k0, k1
    # C block rows i0, i1; cols k0, k1
    def add_strassen(i0, j0, k0):
        # Indices
        # A: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C: (k0, i0), (k0, i1), (k1, i0), (k1, i1)
        
        # M1 = (A00 + A11)(B00 + B11)
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0, j0, 1); set_u(u, i0+1, j0+1, 1)
        set_v(v, j0, k0, 1); set_v(v, j0+1, k0+1, 1)
        # C00 += M1, C11 += M1
        set_w(w, k0, i0, 1)
        set_w(w, k0+1, i0+1, 1)
        add_term(u, v, w)
        
        # M2 = (A10 + A11) B00
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0+1, j0, 1); set_u(u, i0+1, j0+1, 1)
        set_v(v, j0, k0, 1)
        # C10 += M2
        set_w(w, k0, i0+1, 1)
        add_term(u, v, w)
        
        # M3 = A00 (B01 - B11)
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0, j0, 1)
        set_v(v, j0, k0+1, 1); set_v(v, j0+1, k0+1, -1)
        # C01 += M3
        set_w(w, k0+1, i0, 1)
        add_term(u, v, w)
        
        # M4 = A11 (B10 - B00)
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0+1, j0+1, 1)
        set_v(v, j0+1, k0, 1); set_v(v, j0, k0, -1)
        # C10 += M4, C00 += M4
        set_w(w, k0, i0+1, 1)
        set_w(w, k0, i0, 1)
        add_term(u, v, w)
        
        # M5 = (A00 + A01) B11
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0, j0, 1); set_u(u, i0, j0+1, 1)
        set_v(v, j0+1, k0+1, 1)
        # C01 += M5, C00 -= M5
        set_w(w, k0+1, i0, 1)
        set_w(w, k0, i0, -1)
        add_term(u, v, w)
        
        # M6 = (A10 - A00) (B00 + B01)
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0+1, j0, 1); set_u(u, i0, j0, -1)
        set_v(v, j0, k0, 1); set_v(v, j0, k0+1, 1)
        # C11 += M6
        set_w(w, k0+1, i0+1, 1)
        add_term(u, v, w)
        
        # M7 = (A01 - A11) (B10 + B11)
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i0, j0+1, 1); set_u(u, i0+1, j0+1, -1)
        set_v(v, j0+1, k0, 1); set_v(v, j0+1, k0+1, 1)
        # C00 += M7, C11 -= M7
        set_w(w, k0, i0, 1)
        set_w(w, k0+1, i0+1, -1)
        add_term(u, v, w)

    # Schoolbook term A[i,j] B[j,k]
    def add_schoolbook(i, j, k):
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i, j, 1)
        set_v(v, j, k, 1)
        set_w(w, k, i, 1)
        add_term(u, v, w)

    # Top half (i=0,1)
    # A_TL (2x2) B_TL (2x2) -> Strassen
    add_strassen(0, 0, 0)
    
    # A_TL (2x2) B_TR (2x3) -> Schoolbook
    # A rows 0,1; Cols 0,1. B Rows 0,1; Cols 2,3,4.
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
                
    # A_TR (2x3) B_BL (3x2) -> Schoolbook
    # A Rows 0,1; Cols 2,3,4. B Rows 2,3,4; Cols 0,1.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
                
    # A_TR (2x3) B_BR (3x3) -> Schoolbook
    # A Rows 0,1; Cols 2,3,4. B Rows 2,3,4; Cols 2,3,4.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)

    # Bottom half (i=2,3)
    # A_BL (2x2) B_TL (2x2) -> Strassen
    add_strassen(2, 0, 0)
    
    # A_BL (2x2) B_TR (2x3) -> Schoolbook
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
                
    # A_BR (2x3) B_BL (3x2) -> Schoolbook
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
                
    # A_BR (2x3) B_BR (3x3) -> Schoolbook
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
