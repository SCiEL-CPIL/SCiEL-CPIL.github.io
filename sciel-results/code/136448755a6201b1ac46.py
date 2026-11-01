"""Rewritten program for matrix multiplication 2,2,7 with rank 25."""
DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 block
    # Each term is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (i, j, coeff)
    # V_coeffs: list of (j, k_local, coeff)
    # W_coeffs: list of (k_local, i, coeff)
    
    strassen_terms = [
        # T0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # T1: (A10+A11)B00 -> C10+=1, C11-=1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # T2: A00(B01-B11) -> C01+=1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]),
        # T3: A11(B00-B10) -> C00+=1
        ([(1,1,1)], [(0,0,1), (1,0,-1)], [(0,0,1)]),
        # T4: (A00+A01)B11 -> C10+=1, C11+=1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (1,1,1)]),
        # T5: (A10-A00)(B00+B01) -> C00-=1, C01+=1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,-1), (1,0,1)]),
        # T6: (A01-A11)(B10+B11) -> C00-=1, C11+=1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,1,1)])
    ]
    
    col = 0
    # Process 3 blocks of 2 columns each
    for block in range(3):
        k_start = block * 2
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for (i, j, coeff) in u_coeffs:
                U[i * m + j][col] += coeff
                
            for (j, k_loc, coeff) in v_coeffs:
                k_global = k_start + k_loc
                V[j * p + k_global][col] += coeff
                
            for (k_loc, i, coeff) in w_coeffs:
                k_global = k_start + k_loc
                W[k_global * n + i][col] += coeff
                
            col += 1
            
    # Process last single column (k=6) with schoolbook (rank 4)
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # Term 21: A00 B06 -> C06
    U[0 * m + 0][col] = 1
    V[0 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 22: A01 B16 -> C06
    U[0 * m + 1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 23: A10 B06 -> C16
    U[1 * m + 0][col] = 1
    V[0 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    # Term 24: A11 B16 -> C16
    U[1 * m + 1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
