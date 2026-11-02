"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21

    # Strassen terms for each 2x2 block
    # For block b (b=0,1,2), columns c0=2b, c1=2b+1
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B indices within block: 0=B[0,c0], 1=B[0,c1], 2=B[1,c0], 3=B[1,c1]
    # C indices within block: 0=C[0,c0], 1=C[0,c1], 2=C[1,c0], 3=C[1,c1]
    
    # Strassen coefficients: [M1, M2, M3, M4, M5, M6, M7]
    # Each is (A_coeffs, B_coeffs, C_coeffs)
    strassen_terms = [
        # M1: (a+d)(e+h) -> C00+=1, C11+=1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2: (c+d)e -> C10+=1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3: a(f-h) -> C01+=1, C11+=1
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        # M4: d(g-e) -> C00+=1, C10+=1
        ([0,0,0,1], [0,-1,1,0], [1,0,1,0]),
        # M5: (a+b)h -> C00-=1, C01+=1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6: (c-a)(e+f) -> C11+=1
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
        # M7: (b-d)(g+h) -> C00+=1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    for b in range(3):
        c0 = 2 * b
        c1 = 2 * b + 1
        for t in range(7):
            col = b * 7 + t
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
            # U: A indices
            U[0][col] = a_coeffs[0]  # A[0,0]
            U[1][col] = a_coeffs[1]  # A[0,1]
            U[2][col] = a_coeffs[2]  # A[1,0]
            U[3][col] = a_coeffs[3]  # A[1,1]
            # V: B indices
            V[0 * p + c0][col] = b_coeffs[0]  # B[0,c0]
            V[0 * p + c1][col] = b_coeffs[1]  # B[0,c1]
            V[1 * p + c0][col] = b_coeffs[2]  # B[1,c0]
            V[1 * p + c1][col] = b_coeffs[3]  # B[1,c1]
            # W: C indices (k*n + i)
            W[c0 * n + 0][col] = c_coeffs[0]  # C[0,c0]
            W[c1 * n + 0][col] = c_coeffs[1]  # C[0,c1]
            W[c0 * n + 1][col] = c_coeffs[2]  # C[1,c0]
            W[c1 * n + 1][col] = c_coeffs[3]  # C[1,c1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
