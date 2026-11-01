"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28  # 4 blocks * 7 Strassen products
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 multiplication parameters
    # A = [[a, b], [c, d]] -> indices 0, 1, 2, 3
    # B = [[e, f], [g, h]] -> indices depend on block q
    # C = [[c00, c01], [c10, c11]] -> indices depend on block q
    
    # Define the 7 products for a single 2x2 block
    # Each product is defined by (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {index: coeff} for A
    # V_coeffs: dict {index: coeff} for B (local indices 0..3 mapped to global)
    # W_coeffs: dict {index: coeff} for C (local indices 0..3 mapped to global)
    
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_products = [
        # P1: (a+d)(e+h)
        {
            "U": {0: 1, 3: 1},
            "V": {0: 1, 3: 1},
            "W": {0: 1, 3: 1}
        },
        # P2: (c+d)e
        {
            "U": {2: 1, 3: 1},
            "V": {0: 1},
            "W": {2: 1, 3: -1}
        },
        # P3: a(f-h)
        {
            "U": {0: 1},
            "V": {1: 1, 3: -1},
            "W": {1: 1, 3: 1}
        },
        # P4: d(e-f)
        {
            "U": {3: 1},
            "V": {0: 1, 1: -1},
            "W": {0: 1, 2: 1}
        },
        # P5: (a+b)h
        {
            "U": {0: 1, 1: 1},
            "V": {3: 1},
            "W": {0: -1, 1: 1}
        },
        # P6: (c-a)(e+f)
        {
            "U": {2: 1, 0: -1},
            "V": {0: 1, 1: 1},
            "W": {3: 1}
        },
        # P7: (b-d)(g+e)
        {
            "U": {1: 1, 3: -1},
            "V": {2: 1, 0: 1},
            "W": {0: 1}
        }
    ]
    
    col = 0
    for q in range(4):
        for prod in strassen_products:
            # Add U coefficients
            for local_i, coeff in prod["U"].items():
                U[local_i][col] += coeff
            
            # Add V coefficients
            # Local B index lb maps to global B index j*p + k
            # lb 0 -> j=0, k=2q
            # lb 1 -> j=0, k=2q+1
            # lb 2 -> j=1, k=2q
            # lb 3 -> j=1, k=2q+1
            for local_j, coeff in prod["V"].items():
                j = local_j // 2
                k_local = local_j % 2
                k_global = 2 * q + k_local
                global_v_idx = j * p + k_global
                V[global_v_idx][col] += coeff
            
            # Add W coefficients
            # Local C index lc maps to global C index k*n + i
            # lc 0 -> i=0, k=2q
            # lc 1 -> i=0, k=2q+1
            # lc 2 -> i=1, k=2q
            # lc 3 -> i=1, k=2q+1
            for local_k, coeff in prod["W"].items():
                i = local_k % 2
                k_local = local_k // 2
                k_global = 2 * q + k_local
                global_w_idx = k_global * n + i
                W[global_w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
