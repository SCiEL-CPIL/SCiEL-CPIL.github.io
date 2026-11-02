"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 7 Strassen blocks (2x2) + 1 Naive block (2x1)
    # Strassen rank: 7
    # Naive rank for 2x2 by 2x1: 4
    rank = 7 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define Strassen products
    # u: dict (i, j) -> coeff
    # v: dict (j, k) -> coeff (k is local 0 or 1)
    # w: dict (k, i) -> coeff (k is local 0 or 1, i is 0 or 1)
    
    strassen_products = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):-1}),
    ]
    
    # Fill U, V, W for 7 Strassen blocks
    for b in range(7):
        k_start = b * 2
        for prod in strassen_products:
            u_dict, v_dict, w_dict = prod
            
            # U
            for (i, j), val in u_dict.items():
                row_u = i * m + j
                U[row_u][col_idx] += val
                
            # V
            for (j, k_local), val in v_dict.items():
                k_global = k_start + k_local
                row_v = j * p + k_global
                V[row_v][col_idx] += val
                
            # W
            for (k_local, i), val in w_dict.items():
                k_global = k_start + k_local
                row_w = k_global * n + i
                W[row_w][col_idx] += val
                
            col_idx += 1
            
    # Fill U, V, W for 1 Naive block (last column, index 14)
    k_start = 14
    # Naive products:
    # T1: A00 B014 -> C014
    # T2: A01 B114 -> C014
    # T3: A10 B014 -> C114
    # T4: A11 B114 -> C114
    
    naive_products = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}), # A00 B0 -> C0
        ({(0,1):1}, {(1,0):1}, {(0,0):1}), # A01 B1 -> C0
        ({(1,0):1}, {(0,0):1}, {(1,0):1}), # A10 B0 -> C1
        ({(1,1):1}, {(1,0):1}, {(1,0):1}), # A11 B1 -> C1
    ]
    
    for prod in naive_products:
        u_dict, v_dict, w_dict = prod
        
        # U
        for (i, j), val in u_dict.items():
            row_u = i * m + j
            U[row_u][col_idx] += val
            
        # V
        for (j, k_local), val in v_dict.items():
            # k_local is 0 or 1 relative to the 2x1 block?
            # Here B is 2x1, so only k=0 exists in the block.
            # But my naive_products dict uses k_local 0 and 1.
            # For 2x1, B has rows 0,1 and col 0.
            # So B_{0,0} and B_{1,0}.
            # My dict keys are (j, k_local).
            # For T1: B_{0,0}. j=0, k_local=0.
            # For T2: B_{1,0}. j=1, k_local=0.
            # Wait, in naive_products above:
            # T1: v={(0,0):1} -> j=0, k_local=0. Correct.
            # T2: v={(1,0):1} -> j=1, k_local=0. Correct.
            # T3: v={(0,0):1} -> j=0, k_local=0. Correct.
            # T4: v={(1,0):1} -> j=1, k_local=0. Correct.
            k_global = k_start + k_local
            row_v = j * p + k_global
            V[row_v][col_idx] += val
            
        # W
        for (k_local, i), val in w_dict.items():
            k_global = k_start + k_local
            row_w = k_global * n + i
            W[row_w][col_idx] += val
            
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
