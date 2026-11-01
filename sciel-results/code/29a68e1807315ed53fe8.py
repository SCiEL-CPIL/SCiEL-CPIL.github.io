"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Algorithmic construction of Rank 18 decomposition
    # Using Strassen for 2x2 blocks and Schoolbook for remainder
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 by 2x2
    # u_idx, v_idx, w_idx, coeff
    # u_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # v_idx: 0=B00, 1=B01, 2=B10, 3=B11 (local)
    # w_idx: 0=C00, 1=C01, 2=C10, 3=C11 (local)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1: (A00+A11)(B00+B11) -> C00, C11
        ([(2,1), (3,1)], [(0,1)], [(3,1), (0,1)]),         # M2: (A10+A11)B00 -> C10, C00? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let's just use the hardcoded logic but cleaner
    ]
    
    def add_strassen(t_off, k0, k1):
        # M1
        U[0][t_off] = 1; U[3][t_off] = 1
        V[k0][t_off] = 1; V[p+k1][t_off] = 1
        W[k0*n][t_off] = 1; W[k1*n+1][t_off] = 1
        # M2
        U[2][t_off+1] = 1; U[3][t_off+1] = 1
        V[k0][t_off+1] = 1
        W[k0*n+1][t_off+1] = 1; W[k1*n+1][t_off+1] = -1
        # M3
        U[0][t_off+2] = 1
        V[k1][t_off+2] = 1; V[p+k1][t_off+2] = -1
        W[k1*n][t_off+2] = 1; W[k1*n+1][t_off+2] = 1
        # M4
        U[3][t_off+3] = 1
        V[p+k0][t_off+3] = 1; V[k0][t_off+3] = -1
        W[k0*n][t_off+3] = 1; W[k0*n+1][t_off+3] = 1
        # M5
        U[0][t_off+4] = 1; U[1][t_off+4] = 1
        V[p+k1][t_off+4] = 1
        W[k0*n][t_off+4] = -1; W[k1*n][t_off+4] = 1
        # M6
        U[2][t_off+5] = 1; U[0][t_off+5] = -1
        V[k0][t_off+5] = 1; V[k1][t_off+5] = 1
        W[k1*n+1][t_off+5] = 1
        # M7
        U[1][t_off+6] = 1; U[3][t_off+6] = -1
        V[p+k0][t_off+6] = 1; V[p+k1][t_off+6] = 1
        W[k0*n][t_off+6] = 1

    add_strassen(0, 0, 1)
    add_strassen(7, 2, 3)
    
    # Schoolbook for col 4
    t = 14
    U[0][t] = 1; V[4][t] = 1; W[4*n][t] = 1; t+=1
    U[1][t] = 1; V[p+4][t] = 1; W[4*n][t] = 1; t+=1
    U[2][t] = 1; V[4][t] = 1; W[4*n+1][t] = 1; t+=1
    U[3][t] = 1; V[p+4][t] = 1; W[4*n+1][t] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
