import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 3, 7
    
    rank = 39
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen_2x2_2x2(b_col_offset, c_col_start, base_col):
        nonlocal col
        if time.time() > deadline:
            return col
        
        a11_idx = 0 * m + 0
        a12_idx = 0 * m + 1
        a21_idx = 1 * m + 0
        a22_idx = 1 * m + 1
        
        b11_idx = 0 * p + b_col_offset
        b12_idx = 0 * p + b_col_offset + 1
        b21_idx = 1 * p + b_col_offset
        b22_idx = 1 * p + b_col_offset + 1
        
        c11_idx = c_col_start * n + 0
        c12_idx = (c_col_start+1) * n + 0
        c21_idx = c_col_start * n + 1
        c22_idx = (c_col_start+1) * n + 1
        
        # M1 = (A11+A22)(B11+B22)
        U[a11_idx][base_col] += 1
        U[a22_idx][base_col] += 1
        V[b11_idx][base_col] += 1
        V[b22_idx][base_col] += 1
        W[c11_idx][base_col] += 1
        W[c22_idx][base_col] += 1
        col = base_col + 1
        
        # M2 = (A21+A22)B11
        U[a21_idx][col] += 1
        U[a22_idx][col] += 1
        V[b11_idx][col] += 1
        W[c21_idx][col] += 1
        W[c22_idx][col] -= 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[a11_idx][col] += 1
        V[b12_idx][col] += 1
        V[b22_idx][col] -= 1
        W[c12_idx][col] += 1
        W[c22_idx][col] += 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[a22_idx][col] += 1
        V[b21_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c11_idx][col] += 1
        W[c21_idx][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a11_idx][col] += 1
        U[a12_idx][col] += 1
        V[b22_idx][col] += 1
        W[c12_idx][col] += 1
        W[c22_idx][col] -= 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a21_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b11_idx][col] += 1
        V[b12_idx][col] += 1
        W[c22_idx][col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a12_idx][col] += 1
        U[a22_idx][col] -= 1
        V[b21_idx][col] += 1
        V[b22_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        return col
    
    def add_outer_product(b_col_offset, c_col_start, base_col):
        nonlocal col
        if time.time() > deadline:
            return col
        
        a0_idx = 0 * m + 2
        a1_idx = 1 * m + 2
        b0_idx = 2 * p + b_col_offset
        b1_idx = 2 * p + b_col_offset + 1
        
        c00_idx = c_col_start * n + 0
        c01_idx = (c_col_start+1) * n + 0
        c10_idx = c_col_start * n + 1
        c11_idx = (c_col_start+1) * n + 1
        
        U[a0_idx][base_col] += 1
        V[b0_idx][base_col] += 1
        W[c00_idx][base_col] += 1
        col = base_col + 1
        
        U[a0_idx][col] += 1
        V[b1_idx][col] += 1
        W[c01_idx][col] += 1
        col += 1
        
        U[a1_idx][col] += 1
        V[b0_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        U[a1_idx][col] += 1
        V[b1_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        return col
    
    col = add_strassen_2x2_2x2(0, 0, col)
    col = add_outer_product(0, 0, col)
    col = add_strassen_2x2_2x2(2, 2, col)
    col = add_outer_product(2, 2, col)
    col = add_strassen_2x2_2x2(4, 4, col)
    col = add_outer_product(4, 4, col)
    
    if time.time() <= deadline:
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + 6
                c_idx = 6 * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
