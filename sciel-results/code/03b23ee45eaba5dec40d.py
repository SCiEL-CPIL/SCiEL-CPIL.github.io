"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y00)
    # M4 = x11(y11-y01)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def add_strassen_term(A_idx_map, B_idx_map, C_idx_map, col_idx):
        # A_idx_map: dict from 'x00' etc to U index
        # B_idx_map: dict from 'y00' etc to V index
        # C_idx_map: dict from 'Z00' etc to W index
        
        # M1
        U[A_idx_map['x00']][col_idx] += 1
        U[A_idx_map['x11']][col_idx] += 1
        V[B_idx_map['y00']][col_idx] += 1
        V[B_idx_map['y11']][col_idx] += 1
        W[C_idx_map['Z00']][col_idx] += 1
        W[C_idx_map['Z11']][col_idx] += 1
        
        # M2
        U[A_idx_map['x10']][col_idx] += 1
        U[A_idx_map['x11']][col_idx] += 1
        V[B_idx_map['y00']][col_idx] += 1
        W[C_idx_map['Z10']][col_idx] += 1
        W[C_idx_map['Z11']][col_idx] -= 1
        
        # M3
        U[A_idx_map['x00']][col_idx] += 1
        V[B_idx_map['y10']][col_idx] += 1
        V[B_idx_map['y00']][col_idx] -= 1
        W[C_idx_map['Z01']][col_idx] += 1
        W[C_idx_map['Z11']][col_idx] += 1
        
        # M4
        U[A_idx_map['x11']][col_idx] += 1
        V[B_idx_map['y11']][col_idx] += 1
        V[B_idx_map['y01']][col_idx] -= 1
        W[C_idx_map['Z00']][col_idx] += 1
        W[C_idx_map['Z10']][col_idx] += 1
        
        # M5
        U[A_idx_map['x00']][col_idx] += 1
        U[A_idx_map['x01']][col_idx] += 1
        V[B_idx_map['y11']][col_idx] += 1
        W[C_idx_map['Z00']][col_idx] -= 1
        W[C_idx_map['Z01']][col_idx] += 1
        
        # M6
        U[A_idx_map['x10']][col_idx] += 1
        U[A_idx_map['x00']][col_idx] -= 1
        V[B_idx_map['y00']][col_idx] += 1
        V[B_idx_map['y01']][col_idx] += 1
        W[C_idx_map['Z11']][col_idx] += 1
        
        # M7
        U[A_idx_map['x01']][col_idx] += 1
        U[A_idx_map['x11']][col_idx] -= 1
        V[B_idx_map['y10']][col_idx] += 1
        V[B_idx_map['y11']][col_idx] += 1
        W[C_idx_map['Z00']][col_idx] += 1
        
    # Process A_L (cols 0,1) and B_L (rows 0,1)
    A_L_map = {
        'x00': 0, # A[0,0]
        'x01': 1, # A[0,1]
        'x10': 4, # A[1,0]
        'x11': 5  # A[1,1]
    }
    
    for k in range(4):
        # B_L block k: rows 0,1, cols 2k, 2k+1
        B_L_map = {
            'y00': 0 * 8 + 2 * k,
            'y01': 0 * 8 + 2 * k + 1,
            'y10': 1 * 8 + 2 * k,
            'y11': 1 * 8 + 2 * k + 1
        }
        C_map = {
            'Z00': 2 * k * 2 + 0, # C[0, 2k]
            'Z01': (2 * k + 1) * 2 + 0, # C[0, 2k+1]
            'Z10': 2 * k * 2 + 1, # C[1, 2k]
            'Z11': (2 * k + 1) * 2 + 1 # C[1, 2k+1]
        }
        add_strassen_term(A_L_map, B_L_map, C_map, col)
        col += 1
        # Note: Strassen produces 7 terms, but my helper adds 7 sets of coefficients?
        # No, the helper adds the coefficients for ONE bilinear term?
        # Wait. Strassen computes 7 products. Each product is a bilinear term.
        # The helper function above adds coefficients for M1..M7 into the SAME column?
        # No, that's wrong. Each M_i is a separate rank-1 term.
        # So we need 7 columns for each 2x2x2 block.
        
    # Corrected logic:
    # We need to reset col and do it properly.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_one_strassen_product(A_idx_map, B_idx_map, C_idx_map, prod_id, col_idx):
        # prod_id 0..6 corresponds to M1..M7
        
        if prod_id == 0: # M1
            U[A_idx_map['x00']][col_idx] += 1
            U[A_idx_map['x11']][col_idx] += 1
            V[B_idx_map['y00']][col_idx] += 1
            V[B_idx_map['y11']][col_idx] += 1
            W[C_idx_map['Z00']][col_idx] += 1
            W[C_idx_map['Z11']][col_idx] += 1
        elif prod_id == 1: # M2
            U[A_idx_map['x10']][col_idx] += 1
            U[A_idx_map['x11']][col_idx] += 1
            V[B_idx_map['y00']][col_idx] += 1
            W[C_idx_map['Z10']][col_idx] += 1
            W[C_idx_map['Z11']][col_idx] -= 1
        elif prod_id == 2: # M3
            U[A_idx_map['x00']][col_idx] += 1
            V[B_idx_map['y10']][col_idx] += 1
            V[B_idx_map['y00']][col_idx] -= 1
            W[C_idx_map['Z01']][col_idx] += 1
            W[C_idx_map['Z11']][col_idx] += 1
        elif prod_id == 3: # M4
            U[A_idx_map['x11']][col_idx] += 1
            V[B_idx_map['y11']][col_idx] += 1
            V[B_idx_map['y01']][col_idx] -= 1
            W[C_idx_map['Z00']][col_idx] += 1
            W[C_idx_map['Z10']][col_idx] += 1
        elif prod_id == 4: # M5
            U[A_idx_map['x00']][col_idx] += 1
            U[A_idx_map['x01']][col_idx] += 1
            V[B_idx_map['y11']][col_idx] += 1
            W[C_idx_map['Z00']][col_idx] -= 1
            W[C_idx_map['Z01']][col_idx] += 1
        elif prod_id == 5: # M6
            U[A_idx_map['x10']][col_idx] += 1
            U[A_idx_map['x00']][col_idx] -= 1
            V[B_idx_map['y00']][col_idx] += 1
            V[B_idx_map['y01']][col_idx] += 1
            W[C_idx_map['Z11']][col_idx] += 1
        elif prod_id == 6: # M7
            U[A_idx_map['x01']][col_idx] += 1
            U[A_idx_map['x11']][col_idx] -= 1
            V[B_idx_map['y10']][col_idx] += 1
            V[B_idx_map['y11']][col_idx] += 1
            W[C_idx_map['Z00']][col_idx] += 1

    A_L_map = {
        'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5
    }
    A_R_map = {
        'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7
    }
    
    for k in range(4):
        B_L_map = {
            'y00': 0 * 8 + 2 * k, 'y01': 0 * 8 + 2 * k + 1,
            'y10': 1 * 8 + 2 * k, 'y11': 1 * 8 + 2 * k + 1
        }
        B_R_map = {
            'y00': 2 * 8 + 2 * k, 'y01': 2 * 8 + 2 * k + 1,
            'y10': 3 * 8 + 2 * k, 'y11': 3 * 8 + 2 * k + 1
        }
        C_map = {
            'Z00': 2 * k * 2 + 0, 'Z01': (2 * k + 1) * 2 + 0,
            'Z10': 2 * k * 2 + 1, 'Z11': (2 * k + 1) * 2 + 1
        }
        
        for r in range(7):
            add_one_strassen_product(A_L_map, B_L_map, C_map, r, col)
            col += 1
            
        for r in range(7):
            add_one_strassen_product(A_R_map, B_R_map, C_map, r, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
