"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 8 blocks of Strassen 2x2x2
    # Rank = 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[s][i][j], v[s][j'][k], w[s][i'][k']
    # i, j, j', k, i', k' in {0, 1}
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00     -> C10+1, C11-1
    # M3: A00(B01-B11)      -> C00-1, C01+1, C11+1
    # M4: A11(B10-B00)      -> C00+1, C10+1
    # M5: (A00+A01)B11      -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01)-> C11+1
    # M7: (A01-A11)(B10+B11)-> C00+1
    
    # Define non-zero terms for each s
    # Format: (s, u_terms, v_terms, w_terms)
    # u_terms: list of (i, j, val)
    # v_terms: list of (j_prime, k_prime, val)
    # w_terms: list of (i_prime, k_prime, val)
    
    strassen_terms = [
        # s=0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # s=1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # s=2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,0,-1), (0,1,1), (1,1,1)]),
        # s=3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # s=4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # s=5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # s=6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for g in range(8):
        base_k = 2 * g
        for s in range(7):
            r = 7 * g + s
            u_terms, v_terms, w_terms = strassen_terms[s]
            
            for i, j, val in u_terms:
                U[i * m + j][r] += val
                
            for jp, kp, val in v_terms:
                k_global = base_k + kp
                V[jp * p + k_global][r] += val
                
            for ip, kp, val in w_terms:
                k_global = base_k + kp
                W[k_global * n + ip][r] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
