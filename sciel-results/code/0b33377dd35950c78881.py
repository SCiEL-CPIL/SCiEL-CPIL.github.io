import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen M222 block
    # a_rows: list of 2 global row indices for A (i)
    # a_cols: list of 2 global col indices for A (j)
    # b_rows: list of 2 global row indices for B (j)
    # b_cols: list of 2 global col indices for B (k)
    # c_rows: list of 2 global row indices for C (i)
    # c_cols: list of 2 global col indices for C (k)
    # Note: A is n x m, so index is i*m + j
    # B is m x p, so index is j*p + k
    # C is p x n, so index is k*n + i (Trace convention W[k*n+i, r])
    # Wait, the task says W[k*n+i, r].
    # C is 2x5. i is row (0..1), k is col (0..4).
    # Index in W: k*n + i = k*2 + i.
    
    def add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # Strassen formulas for C = A B
        # A is 2x2, B is 2x2
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        
        # Term 1: M1
        # U: A00+A11
        U[a_i[0]*m + a_j[0], col] = 1
        U[a_i[1]*m + a_j[1], col] = 1
        # V: B00+B11
        V[b_j[0]*p + b_k[0], col] = 1
        V[b_j[1]*p + b_k[1], col] = 1
        # W: C00 + C11
        W[c_k[0]*n + c_i[0], col] = 1
        W[c_k[1]*n + c_i[1], col] = 1
        col += 1
        
        # Term 2: M2
        # U: A10+A11
        U[a_i[1]*m + a_j[0], col] = 1
        U[a_i[1]*m + a_j[1], col] = 1
        # V: B00
        V[b_j[0]*p + b_k[0], col] = 1
        # W: C10 - C11
        W[c_k[0]*n + c_i[1], col] = 1
        W[c_k[1]*n + c_i[1], col] = -1
        col += 1
        
        # Term 3: M3
        # U: A00
        U[a_i[0]*m + a_j[0], col] = 1
        # V: B10-B11
        V[b_j[1]*p + b_k[0], col] = 1
        V[b_j[1]*p + b_k[1], col] = -1
        # W: C01 + C11
        W[c_k[1]*n + c_i[0], col] = 1
        W[c_k[1]*n + c_i[1], col] = 1
        col += 1
        
        # Term 4: M4
        # U: A11
        U[a_i[1]*m + a_j[1], col] = 1
        # V: B01-B00
        V[b_j[0]*p + b_k[1], col] = 1
        V[b_j[0]*p + b_k[0], col] = -1
        # W: C00 + C10
        W[c_k[0]*n + c_i[0], col] = 1
        W[c_k[0]*n + c_i[1], col] = 1
        col += 1
        
        # Term 5: M5
        # U: A00+A01
        U[a_i[0]*m + a_j[0], col] = 1
        U[a_i[0]*m + a_j[1], col] = 1
        # V: B11
        V[b_j[1]*p + b_k[1], col] = 1
        # W: -C00 + C01
        W[c_k[0]*n + c_i[0], col] = -1
        W[c_k[1]*n + c_i[0], col] = 1
        col += 1
        
        # Term 6: M6
        # U: A10-A00
        U[a_i[1]*m + a_j[0], col] = 1
        U[a_i[0]*m + a_j[0], col] = -1
        # V: B00+B01
        V[b_j[0]*p + b_k[0], col] = 1
        V[b_j[0]*p + b_k[1], col] = 1
        # W: C11
        W[c_k[1]*n + c_i[1], col] = 1
        col += 1
        
        # Term 7: M7
        # U: A01-A11
        U[a_i[0]*m + a_j[1], col] = 1
        U[a_i[1]*m + a_j[1], col] = -1
        # V: B10+B11
        V[b_j[1]*p + b_k[0], col] = 1
        V[b_j[1]*p + b_k[1], col] = 1
        # W: C00
        W[c_k[0]*n + c_i[0], col] = 1
        col += 1

    def add_standard_221(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # C = A B, A is 2x2, B is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # 4 terms
        # T1: A00 B0 -> C0
        U[a_i[0]*m + a_j[0], col] = 1
        V[b_j[0]*p + b_k[0], col] = 1
        W[c_k[0]*n + c_i[0], col] = 1
        col += 1
        
        # T2: A01 B1 -> C0
        U[a_i[0]*m + a_j[1], col] = 1
        V[b_j[1]*p + b_k[0], col] = 1
        W[c_k[0]*n + c_i[0], col] = 1
        col += 1
        
        # T3: A10 B0 -> C1
        U[a_i[1]*m + a_j[0], col] = 1
        V[b_j[0]*p + b_k[0], col] = 1
        W[c_k[0]*n + c_i[1], col] = 1
        col += 1
        
        # T4: A11 B1 -> C1
        U[a_i[1]*m + a_j[1], col] = 1
        V[b_j[1]*p + b_k[0], col] = 1
        W[c_k[0]*n + c_i[1], col] = 1
        col += 1

    def add_standard_215(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # C = A B, A is 2x1, B is 1x5
        # C_i,k = A_i B_k
        # 10 terms
        for i_idx in range(2):
            for k_idx in range(5):
                U[a_i[i_idx]*m + a_j[0], col] = 1
                V[b_j[0]*p + b_k[k_idx], col] = 1
                W[c_k[k_idx]*n + c_i[i_idx], col] = 1
                col += 1

    # Block 1: j in {0, 1}
    a_i = [0, 1]
    a_j = [0, 1]
    b_j = [0, 1]
    
    # Sub-block 1a: k in {0, 1}
    b_k = [0, 1]
    c_i = [0, 1]
    c_k = [0, 1]
    add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Sub-block 1b: k in {2, 3}
    b_k = [2, 3]
    c_k = [2, 3]
    add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Sub-block 1c: k in {4}
    b_k = [4]
    c_k = [4]
    add_standard_221(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Block 2: j in {2, 3}
    a_j = [2, 3]
    b_j = [2, 3]
    
    # Sub-block 2a: k in {0, 1}
    b_k = [0, 1]
    c_k = [0, 1]
    add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Sub-block 2b: k in {2, 3}
    b_k = [2, 3]
    c_k = [2, 3]
    add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Sub-block 2c: k in {4}
    b_k = [4]
    c_k = [4]
    add_standard_221(a_i, a_j, b_j, b_k, c_i, c_k)
    
    # Block 3: j in {4}
    a_j = [4]
    b_j = [4]
    b_k = [0, 1, 2, 3, 4]
    c_k = [0, 1, 2, 3, 4]
    add_standard_215(a_i, a_j, b_j, b_k, c_i, c_k)
    
    return {"U": U, "V": V, "W": W}
