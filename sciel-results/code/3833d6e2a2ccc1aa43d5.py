def _construct_rank_26():
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Standard multiplication for terms involving index 2 (19 terms)
    # These are all (i,j,k) where at least one of i,j,k is 2
    standard_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    standard_terms.append((i, j, k))
    
    for i, j, k in standard_terms:
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    # Strassen 2x2 core (i,j in {0,1}, k in {0,1})
    # 7 terms instead of 4 standard terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a product of a linear form in A and a linear form in B.
    # M1 = (A00+A11)(B00+B11)
    # This contributes to C00, C01, C10, C11 with coefficients 1, 0, 0, 1? No...
    
    # I think the issue is that Strassen's algorithm computes the 2x2 product using 7 scalar
    # multiplications, but each scalar multiplication is of the form (linear_combination_of_A) * 
    # (linear_combination_of_B), and the result is combined with linear combinations to get C.
    
    # In tensor decomposition, each rank-1 term corresponds to one such scalar multiplication.
    # So M1 = (A00+A11)(B00+B11) is ONE rank-1 term where:
    # U has 1 at positions corresponding to A00 and A11
    # V has 1 at positions corresponding to B00 and B11
    # W has coefficients that distribute M1 to the appropriate C entries
    
    # For M1:
    # U[0, col] = 1 (A00)
    # U[4, col] = 1 (A11)
    # V[0, col] = 1 (B00)
    # V[3, col] = 1 (B11)
    # W[0, col] = 1 (contributes to C00)
    # W[1, col] = 0
    # W[3, col] = 1 (contributes to C11)
    # W[4, col] = 0
    
    # Wait, W[k*n+i, r] corresponds to C[i,k]. So:
    # C00: W[0*3+0] = W[0]
    # C01: W[1*3+0] = W[3]
    # C10: W[0*3+1] = W[1]
    # C11: W[1*3+1] = W[4]
    
    # M1 contributes to C00 and C11 with coefficient 1 each.
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # Contributes to C10 with coefficient 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    # Contributes to C01 with coefficient 1
    U[0][col] = 1
    V[3][col] = 1; V[6][col] = -1
    W[3][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    # Contributes to C00 with coeff 1, C10 with coeff 1
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # Contributes to C00 with coeff -1, C01 with coeff 1
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # Contributes to C11 with coeff 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # Contributes to C00 with coeff 1
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    else:
        return None
