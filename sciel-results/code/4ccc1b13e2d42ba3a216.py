import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 3 k-blocks * (7 Strassen + 7 Strassen + 4 Schoolbook) = 3 * 18 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block (2x2x2)
    def add_strassen(i_off, j_off, k_off, col_start):
        # Local indices 0,1
        # m1 = (a+d)(e+h)
        # U: a(0,0), d(1,1) -> U[0], U[3]
        # V: e(0,0), h(1,1) -> V[0], V[3]
        # W: C00(0,0), C01(0,1), C10(1,0), C11(1,1) -> W[0], W[1], W[2], W[3]
        
        def set_U(i, j, val):
            U[i_off + i][j_off + j] # Wait, U is list of lists.
            # U index is i*m + j.
            # But we are filling U[row][col].
            # U[row] is a list of length rank.
            # So U[i*m + j][col] = val.
            pass
        
        # We need to modify U, V, W directly.
        # U is shape (n*m, rank).
        # V is shape (m*p, rank).
        # W is shape (p*n, rank).
        
        # m1
        U[(i_off + 0) * m + (j_off + 0)][col_start] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 0)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 1)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 0)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # m2 = (c+d)e
        # U: c(1,0), d(1,1)
        # V: e(0,0)
        # W: C10 (k=0, i=1) -> W[1]
        U[(i_off + 1) * m + (j_off + 0)][col_start] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # m3 = a(f-h)
        # U: a(0,0)
        # V: f(0,1), h(1,1)
        # W: C01 (k=1, i=0) -> W[2], C11 (k=1, i=1) -> W[3]
        U[(i_off + 0) * m + (j_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 1)][col_start] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_start] = -1
        W[(k_off + 1) * n + (i_off + 0)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # m4 = d(g-e)
        # U: d(1,1)
        # V: g(1,0), e(0,0)
        # W: C10 (k=0, i=1) -> W[1], C11 (k=1, i=1) -> W[3]
        U[(i_off + 1) * m + (j_off + 1)][col_start] = 1
        V[(j_off + 1) * p + (k_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = -1
        W[(k_off + 0) * n + (i_off + 1)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # m5 = (a+b)h
        # U: a(0,0), b(0,1)
        # V: h(1,1)
        # W: C00 (k=0, i=0) -> W[0], C01 (k=1, i=0) -> W[2]
        U[(i_off + 0) * m + (j_off + 0)][col_start] = 1
        U[(i_off + 0) * m + (j_off + 1)][col_start] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 0)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 0)][col_start] = 1
        col_start += 1
        
        # m6 = (c-a)(e+f)
        # U: c(1,0), a(0,0)
        # V: e(0,0), f(0,1)
        # W: C11 (k=1, i=1) -> W[3]
        U[(i_off + 1) * m + (j_off + 0)][col_start] = 1
        U[(i_off + 0) * m + (j_off + 0)][col_start] = -1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # m7 = (b-d)(g+h)
        # U: b(0,1), d(1,1)
        # V: g(1,0), h(1,1)
        # W: C00 (k=0, i=0) -> W[0], C11 (k=1, i=1) -> W[3]
        U[(i_off + 0) * m + (j_off + 1)][col_start] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_start] = -1
        V[(j_off + 1) * p + (k_off + 0)][col_start] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 0)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        return col_start

    # Helper to add a Schoolbook 2x1x2 block
    def add_schoolbook_212(i_off, j_off, k_off, col_start):
        # j is single index (j_off)
        # i in 0,1; k in 0,1
        # 4 terms
        # i=0, k=0
        U[(i_off + 0) * m + (j_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 0)][col_start] = 1
        col_start += 1
        
        # i=0, k=1
        U[(i_off + 0) * m + (j_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 0)][col_start] = 1
        col_start += 1
        
        # i=1, k=0
        U[(i_off + 1) * m + (j_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 0)][col_start] = 1
        W[(k_off + 0) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        # i=1, k=1
        U[(i_off + 1) * m + (j_off + 0)][col_start] = 1
        V[(j_off + 0) * p + (k_off + 1)][col_start] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_start] = 1
        col_start += 1
        
        return col_start

    for t in range(3):
        k_off = 2 * t
        # Part 1: j in 0,1
        col = add_strassen(0, 0, k_off, col)
        # Part 2a: j in 2,3
        col = add_strassen(0, 2, k_off, col)
        # Part 2b: j in 4
        col = add_schoolbook_212(0, 4, k_off, col)

    return {"U": U, "V": V, "W": W}
