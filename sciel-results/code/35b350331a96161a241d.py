import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i in range(n * m):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(m * p):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(p * n):
            if w[i] != 0:
                W[i][col] = w[i]
        col += 1
    
    # Strassen for 2x2x2
    def add_strassen(i_start, j_start, k_start):
        # A indices (2x2 block starting at row i_start, col j_start)
        a00 = i_start * m + j_start
        a01 = i_start * m + (j_start + 1)
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + (j_start + 1)
        
        # B indices (2x2 block starting at row j_start, col k_start)
        b00 = j_start * p + k_start
        b01 = j_start * p + (k_start + 1)
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + (k_start + 1)
        
        # C indices (2x2 block: rows i_start, i_start+1; cols k_start, k_start+1)
        # W index: k*n + i
        c00 = k_start * n + i_start
        c01 = (k_start + 1) * n + i_start
        c10 = k_start * n + (i_start + 1)
        c11 = (k_start + 1) * n + (i_start + 1)
        
        # m1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a11] = 1
        v[b00] = 1; v[b11] = 1
        w[c00] = 1; w[c11] = 1
        add_term(u, v, w)
        
        # m2 = (a10+a11)b00 -> c10 += 1, c11 -= 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a11] = 1
        v[b00] = 1
        w[c10] = 1; w[c11] = -1
        add_term(u, v, w)
        
        # m3 = a00(b01-b11) -> c01 += 1, c11 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1
        v[b01] = 1; v[b11] = -1
        w[c01] = 1; w[c11] = 1
        add_term(u, v, w)
        
        # m4 = a11(b10-b00) -> c00 += 1, c10 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a11] = 1
        v[b10] = 1; v[b00] = -1
        w[c00] = 1; w[c10] = 1
        add_term(u, v, w)
        
        # m5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a01] = 1
        v[b11] = 1
        w[c00] = -1; w[c01] = 1
        add_term(u, v, w)
        
        # m6 = (a10-a00)(b00+b01) -> c11 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a00] = -1
        v[b00] = 1; v[b01] = 1
        w[c11] = 1
        add_term(u, v, w)
        
        # m7 = (a01-a11)(b10+b11) -> c00 += 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a01] = 1; u[a11] = -1
        v[b10] = 1; v[b11] = 1
        w[c00] = 1
        add_term(u, v, w)
    
    # Schoolbook for 1x2x2
    def add_1x2x2(i_row, j_start, k_start):
        a0 = i_row * m + j_start
        a1 = i_row * m + (j_start + 1)
        
        b00 = j_start * p + k_start
        b01 = j_start * p + (k_start + 1)
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + (k_start + 1)
        
        c0 = k_start * n + i_row
        c1 = (k_start + 1) * n + i_row
        
        # x1*y1 -> z1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0] = 1; v[b00] = 1; w[c0] = 1
        add_term(u, v, w)
        
        # x2*y3 -> z1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1] = 1; v[b10] = 1; w[c0] = 1
        add_term(u, v, w)
        
        # x1*y2 -> z2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a0] = 1; v[b01] = 1; w[c1] = 1
        add_term(u, v, w)
        
        # x2*y4 -> z2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a1] = 1; v[b11] = 1; w[c1] = 1
        add_term(u, v, w)
    
    # Main: 8 Strassen blocks (rows 1,2) + 8 schoolbook blocks (row 0)
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            add_strassen(1, j_start, k_start)
            add_1x2x2(0, j_start, k_start)
    
    return {"U": U, "V": V, "W": W}
