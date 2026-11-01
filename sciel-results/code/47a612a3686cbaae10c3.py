"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    The decomposition is Strassen's algorithm, which is rank-minimal for 2x2
    matrix multiplication over the reals (proven lower bound of 7).
    
    Indexing convention:
      U has shape (n*m, rank) = (4, 7); row index = i*m + j  (A[i,j])
      V has shape (m*p, rank) = (4, 7); row index = j*p + k  (B[j,k])
      W has shape (p*n, rank) = (4, 7); row index = k*n + i  (C[i,k], trace convention)
    
    The tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products with their factor coefficients.
    # For each product r: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs maps A-index (i*m+j) to coefficient
    # V_coeffs maps B-index (j*p+k) to coefficient  
    # W_coeffs maps C-index (k*n+i) to coefficient
    #
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices (W[k*n+i]): C00(k=0,i=0)=0, C10(k=0,i=1)=1, C01(k=1,i=0)=2, C11(k=1,i=1)=3
    
    strassen_products = [
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        
        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        
        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for r, (u_c, v_c, w_c) in enumerate(strassen_products):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
