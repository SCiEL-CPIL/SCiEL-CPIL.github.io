"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: A[i][j'], B[j'][k'], C[i][k']
    # We define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(i, j'): val}
    # B_coeffs: dict {(j', k'): val}
    # C_coeffs: dict {(i, k'): val}
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for b_j in range(2):
        for b_k in range(3):
            for A_coeffs, B_coeffs, C_coeffs in strassen_terms:
                # Process A coefficients
                for (i, j_prime), val in A_coeffs.items():
                    j_global = 2 * b_j + j_prime
                    u_idx = i * m + j_global
                    U[u_idx][col] += val
                    
                # Process B coefficients
                for (j_prime, k_prime), val in B_coeffs.items():
                    j_global = 2 * b_j + j_prime
                    k_global = 2 * b_k + k_prime
                    v_idx = j_global * p + k_global
                    V[v_idx][col] += val
                    
                # Process C coefficients
                for (i, k_prime), val in C_coeffs.items():
                    k_global = 2 * b_k + k_prime
                    w_idx = k_global * n + i
                    W[w_idx][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
