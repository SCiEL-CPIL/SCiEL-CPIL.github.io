"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(r1, r2, c1, c2, k1, k2):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[r1 * m + c1][col] += 1
        U[r2 * m + c2][col] += 1
        V[c1 * p + k1][col] += 1
        V[c1 * p + k2][col] += 1
        V[c2 * p + k1][col] += 1
        V[c2 * p + k2][col] += 1
        W[k1 * n + r1][col] += 1
        W[k2 * n + r2][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[r1 * m + c2][col] += 1
        U[r2 * m + c2][col] += 1
        V[c1 * p + k1][col] += 1
        W[k1 * n + r2][col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[r1 * m + c1][col] += 1
        V[c1 * p + k2][col] += 1
        V[c2 * p + k2][col] -= 1
        W[k2 * n + r1][col] += 1
        W[k2 * n + r2][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[r2 * m + c2][col] += 1
        V[c2 * p + k1][col] += 1
        V[c1 * p + k1][col] -= 1
        W[k1 * n + r1][col] += 1
        W[k1 * n + r2][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[r1 * m + c1][col] += 1
        U[r1 * m + c2][col] += 1
        V[c2 * p + k2][col] += 1
        W[k2 * n + r1][col] += 1
        W[k1 * n + r1][col] -= 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[r1 * m + c2][col] += 1
        U[r1 * m + c1][col] -= 1
        V[c1 * p + k1][col] += 1
        V[c1 * p + k2][col] += 1
        W[k1 * n + r1][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[r1 * m + c2][col] += 1
        U[r2 * m + c2][col] -= 1
        V[c2 * p + k1][col] += 1
        V[c2 * p + k2][col] += 1
        W[k2 * n + r2][col] += 1
        col += 1

    def add_naive_1x2x2(r1, c1, c2, k1, k2):
        nonlocal col
        # x1 y11 -> z1
        U[r1 * m + c1][col] += 1
        V[c1 * p + k1][col] += 1
        W[k1 * n + r1][col] += 1
        col += 1
        
        # x2 y21 -> z1
        U[r1 * m + c2][col] += 1
        V[c2 * p + k1][col] += 1
        W[k1 * n + r1][col] += 1
        col += 1
        
        # x1 y12 -> z2
        U[r1 * m + c1][col] += 1
        V[c1 * p + k2][col] += 1
        W[k2 * n + r1][col] += 1
        col += 1
        
        # x2 y22 -> z2
        U[r1 * m + c2][col] += 1
        V[c2 * p + k2][col] += 1
        W[k2 * n + r1][col] += 1
        col += 1

    # A1 (rows 0,1) @ B1 (cols 0-3)
    # A1a (cols 0,1) @ B11 (rows 0,1, cols 0,1) -> C[0,1,0,1]
    add_strassen(0, 1, 0, 1, 0, 1)
    # A1b (cols 2,3) @ B13 (rows 2,3, cols 0,1) -> C[0,1,0,1]
    add_strassen(0, 1, 2, 3, 0, 1)
    # A1a (cols 0,1) @ B12 (rows 0,1, cols 2,3) -> C[0,1,2,3]
    add_strassen(0, 1, 0, 1, 2, 3)
    # A1b (cols 2,3) @ B14 (rows 2,3, cols 2,3) -> C[0,1,2,3]
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # A1 (rows 0,1) @ B2 (cols 4-7)
    # A1a (cols 0,1) @ B21 (rows 0,1, cols 4,5) -> C[0,1,4,5]
    add_strassen(0, 1, 0, 1, 4, 5)
    # A1b (cols 2,3) @ B23 (rows 2,3, cols 4,5) -> C[0,1,4,5]
    add_strassen(0, 1, 2, 3, 4, 5)
    # A1a (cols 0,1) @ B22 (rows 0,1, cols 6,7) -> C[0,1,6,7]
    add_strassen(0, 1, 0, 1, 6, 7)
    # A1b (cols 2,3) @ B24 (rows 2,3, cols 6,7) -> C[0,1,6,7]
    add_strassen(0, 1, 2, 3, 6, 7)
    
    # A2 (row 2) @ B1 (cols 0-3)
    # A2a (cols 0,1) @ B11 (rows 0,1, cols 0,1) -> C[2,0,1]
    add_naive_1x2x2(2, 0, 1, 0, 1)
    # A2b (cols 2,3) @ B13 (rows 2,3, cols 0,1) -> C[2,0,1]
    add_naive_1x2x2(2, 2, 3, 0, 1)
    # A2a (cols 0,1) @ B12 (rows 0,1, cols 2,3) -> C[2,2,3]
    add_naive_1x2x2(2, 0, 1, 2, 3)
    # A2b (cols 2,3) @ B14 (rows 2,3, cols 2,3) -> C[2,2,3]
    add_naive_1x2x2(2, 2, 3, 2, 3)
    
    # A2 (row 2) @ B2 (cols 4-7)
    # A2a (cols 0,1) @ B21 (rows 0,1, cols 4,5) -> C[2,4,5]
    add_naive_1x2x2(2, 0, 1, 4, 5)
    # A2b (cols 2,3) @ B23 (rows 2,3, cols 4,5) -> C[2,4,5]
    add_naive_1x2x2(2, 2, 3, 4, 5)
    # A2a (cols 0,1) @ B22 (rows 0,1, cols 6,7) -> C[2,6,7]
    add_naive_1x2x2(2, 0, 1, 6, 7)
    # A2b (cols 2,3) @ B24 (rows 2,3, cols 6,7) -> C[2,6,7]
    add_naive_1x2x2(2, 2, 3, 6, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
