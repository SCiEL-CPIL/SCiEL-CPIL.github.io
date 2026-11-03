import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: dict or list of (idx, val) for A indices (i*m+j)
        v_vec: dict or list of (idx, val) for B indices (j*p+k)
        w_vec: dict or list of (idx, val) for C indices (k*n+i)
        """
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        
        for idx, val in u_vec:
            u[idx] = val
        for idx, val in v_vec:
            v[idx] = val
        for idx, val in w_vec:
            w[idx] = val
            
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # 1. Strassen blocks for i in {0,1}, j in {0,1}
    # k pairs: (0,1), (2,3), (4,5), (6,7)
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for k0, k1 in k_pairs:
        # Indices
        i0, i1 = 0, 1
        j0, j1 = 0, 1
        
        # A indices
        A00 = i0 * m + j0
        A01 = i0 * m + j1
        A10 = i1 * m + j0
        A11 = i1 * m + j1
        
        # B indices
        B00 = j0 * p + k0
        B01 = j0 * p + k1
        B10 = j1 * p + k0
        B11 = j1 * p + k1
        
        # C indices
        C00 = k0 * n + i0
        C01 = k1 * n + i0
        C10 = k0 * n + i1
        C11 = k1 * n + i1
        
        # 7 Strassen products
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(
            [(A00, 1), (A11, 1)],
            [(B00, 1), (B11, 1)],
            [(C00, 1), (C11, 1)]
        )
        
        # M2 = (A10 + A11) * B00
        add_term(
            [(A10, 1), (A11, 1)],
            [(B00, 1)],
            [(C10, 1)]
        )
        
        # M3 = A00 * (B01 - B11)
        add_term(
            [(A00, 1)],
            [(B01, 1), (B11, -1)],
            [(C01, 1)]
        )
        
        # M4 = A11 * (B00 - B01)
        add_term(
            [(A11, 1)],
            [(B00, 1), (B01, -1)],
            [(C00, 1), (C10, 1)]
        )
        
        # M5 = (A00 + A01) * B11
        add_term(
            [(A00, 1), (A01, 1)],
            [(B11, 1)],
            [(C00, -1), (C01, 1)]
        )
        
        # M6 = (A10 - A00) * (B00 + B10)
        add_term(
            [(A10, 1), (A00, -1)],
            [(B00, 1), (B10, 1)],
            [(C11, 1)]
        )
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(
            [(A01, 1), (A11, -1)],
            [(B10, 1), (B11, 1)],
            [(C00, 1), (C11, -1)]
        )
        
    # 2. Column 2 terms: A[i, 2] * B[2, k] for i in {0,1,2}, k in {0..7}
    for i in range(n):
        for k in range(p):
            A_idx = i * m + 2
            B_idx = 2 * p + k
            C_idx = k * n + i
            add_term(
                [(A_idx, 1)],
                [(B_idx, 1)],
                [(C_idx, 1)]
            )
            
    # 3. Row 2 terms: A[2, j] * B[j, k] for j in {0,1}, k in {0..7}
    for j in range(2):
        for k in range(p):
            A_idx = 2 * m + j
            B_idx = j * p + k
            C_idx = k * n + 2
            add_term(
                [(A_idx, 1)],
                [(B_idx, 1)],
                [(C_idx, 1)]
            )
            
    rank = len(U_cols)
    U = [list(col) for col in U_cols]
    V = [list(col) for col in V_cols]
    W = [list(col) for col in W_cols]
    
    # Transpose to match expected shape?
    # Interface: U shape (n*m, rank). My U_cols is list of rank vectors of length n*m.
    # So U should be a list of rows? Or list of columns?
    # "factor matrices of shapes (n*m, rank)"
    # Usually this means U[i, r].
    # My U_cols[r] is a vector of length n*m.
    # So U[i] should be [U_cols[0][i], U_cols[1][i], ...]
    
    U_final = [[0]*rank for _ in range(n*m)]
    V_final = [[0]*rank for _ in range(m*p)]
    W_final = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_final[i][r] = U_cols[r][i]
        for j in range(m*p):
            V_final[j][r] = V_cols[r][j]
        for k in range(p*n):
            W_final[k][r] = W_cols[r][k]
            
    return {"U": U_final, "V": V_final, "W": W_final}
