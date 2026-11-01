"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-21 decomposition for 2x2 by 2x6 matrix multiplication
    using Strassen's algorithm applied to three 2x2 blocks of the second matrix.
    """
    n, m, p = DIMENSIONS
    # Rank is 7 * (p/2) = 7 * 3 = 21
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # U indices: 0: A00, 1: A01, 2: A10, 3: A11
    # V local indices (for a 2x2 block): 0: B00, 1: B01, 2: B10, 3: B11
    # W local indices (for a 2x2 block result): 0: C00, 1: C01, 2: C10, 3: C11
    
    # Define the 7 products (u_coeff, v_coeff, w_contributions)
    # w_contributions is a list of (local_w_idx, coeff)
    products = [
        # M1: (A00+A11)(B00+B11)
        {
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {0: 1, 3: 1} # C00, C11
        },
        # M2: (A10+A11)(B00+B11)
        {
            'u': {2: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {2: 1, 3: -1} # C10, C11
        },
        # M3: A00(B01-B10)
        {
            'u': {0: 1},
            'v': {1: 1, 2: -1},
            'w': {1: 1, 3: 1} # C01, C11
        },
        # M4: A11(B10-B00)
        {
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {0: 1, 2: 1} # C00, C10
        },
        # M5: (A00+A01)(B11-B10)
        {
            'u': {0: 1, 1: 1},
            'v': {3: 1, 2: -1},
            'w': {0: -1, 1: 1} # C00, C01
        },
        # M6: (A10+A00)(B00+B01)
        {
            'u': {2: 1, 0: 1},
            'v': {0: 1, 1: 1},
            'w': {3: 1} # C11
        },
        # M7: (A01+A11)(B01-B11)
        {
            'u': {1: 1, 3: 1},
            'v': {1: 1, 3: -1},
            'w': {0: 1} # C00
        }
    ]
    
    # Map local indices to global indices
    # U global index: i*m + j. i is row of A, j is col of A.
    # u_idx 0 -> A00 (i=0, j=0) -> 0*2+0 = 0
    # u_idx 1 -> A01 (i=0, j=1) -> 0*2+1 = 1
    # u_idx 2 -> A10 (i=1, j=0) -> 1*2+0 = 2
    # u_idx 3 -> A11 (i=1, j=1) -> 1*2+1 = 3
    u_map = [0, 1, 2, 3]
    
    # V global index: j*p + k. j is row of B, k is col of B.
    # For block q (0, 1, 2), cols are 2q, 2q+1.
    # v_idx 0 -> B00 (j=0, k=2q) -> 0*6 + 2q = 2q
    # v_idx 1 -> B01 (j=0, k=2q+1) -> 0*6 + 2q+1 = 2q+1
    # v_idx 2 -> B10 (j=1, k=2q) -> 1*6 + 2q = 6+2q
    # v_idx 3 -> B11 (j=1, k=2q+1) -> 1*6 + 2q+1 = 7+2q
    
    # W global index: k*n + i. k is col of C, i is row of C.
    # For block q, cols are 2q, 2q+1.
    # w_idx 0 -> C00 (k=2q, i=0) -> 2q*2 + 0 = 4q
    # w_idx 1 -> C01 (k=2q+1, i=0) -> (2q+1)*2 + 0 = 4q+2
    # w_idx 2 -> C10 (k=2q, i=1) -> 2q*2 + 1 = 4q+1
    # w_idx 3 -> C11 (k=2q+1, i=1) -> (2q+1)*2 + 1 = 4q+3
    # Note: My previous manual check used:
    # C00: 4q
    # C01: 4q+2
    # C10: 4q+1
    # C11: 4q+3
    # Let's verify W index formula: k*n + i.
    # C00: k=2q, i=0 -> 2q*2+0 = 4q. Correct.
    # C01: k=2q+1, i=0 -> (2q+1)*2+0 = 4q+2. Correct.
    # C10: k=2q, i=1 -> 2q*2+1 = 4q+1. Correct.
    # C11: k=2q+1, i=1 -> (2q+1)*2+1 = 4q+3. Correct.
    
    w_map_base = [0, 2, 1, 3] # local w_idx 0,1,2,3 map to offsets 0, 2, 1, 3 relative to 4q?
    # local 0 (C00) -> offset 0
    # local 1 (C01) -> offset 2
    # local 2 (C10) -> offset 1
    # local 3 (C11) -> offset 3
    
    for q in range(3):
        col_offset = 7 * q
        # V indices for this block
        v_map = [
            2*q,       # B00
            2*q + 1,   # B01
            6 + 2*q,   # B10
            7 + 2*q    # B11
        ]
        
        # W indices for this block
        w_base = 4 * q
        w_map = [
            w_base + 0, # C00
            w_base + 2, # C01
            w_base + 1, # C10
            w_base + 3  # C11
        ]
        
        for r, prod in enumerate(products):
            col = col_offset + r
            
            # Fill U
            for local_u, val in prod['u'].items():
                global_u = u_map[local_u]
                U[global_u][col] = val
                
            # Fill V
            for local_v, val in prod['v'].items():
                global_v = v_map[local_v]
                V[global_v][col] = val
                
            # Fill W
            for local_w, val in prod['w'].items():
                global_w = w_map[local_w]
                W[global_w][col] = val
                
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W):
    """
    Verifies the decomposition: U * V^T * W^T should equal the multiplication tensor.
    T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    Here indices are flattened:
    i -> (row_A, col_A)
    j -> (row_B, col_B)
    k -> (row_C, col_C) -> actually (col_B, row_C) based on W[k*n+i]
    
    The tensor T is defined such that C_{i, k} = sum_j A_{i, j} B_{j, k}.
    In the decomposition:
    sum_c U[row_A, col_A, c] * V[row_B, col_B, c] * W[col_B, row_C, c] = delta(row_C == row_A) * delta(col_B == col_B) * delta(col_A == row_B)?
    
    Actually, the standard check is:
    For all i (row A), j (col A / row B), k (col B):
    The coefficient of A_{i,j} B_{j,k} in the product should be 1 if it contributes to C_{i,k}.
    
    The decomposition computes:
    C_{i, k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) * W_{k,i,c} ?
    No, the structure is:
    U is (n*m) x rank. Rows are (i,j).
    V is (m*p) x rank. Rows are (j,k).
    W is (p*n) x rank. Rows are (k,i).
    
    The computed value for C_{i, k} is:
    sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_l V_{l,k,c} B_{l,k})
    
    We want this to equal sum_j A_{i,j} B_{j,k}.
    
    So for a specific (i, j, k), the term A_{i,j} B_{j,k} should appear with coefficient 1.
    This requires that for each c, the combination of U, V, W factors out the A and B terms correctly.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor T from U, V, W
    # T[i, j, k] is the coefficient of A_{i,j} B_{j,k} in the result.
    # The result C_{i, k} = sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_l V_{l,k,c} B_{l,k})
    # Expanding:
    # C_{i, k} = sum_j sum_l (sum_c W_{k,i,c} U_{i,j,c} V_{l,k,c}) A_{i,j} B_{l,k}
    # We want C_{i, k} = sum_j A_{i,j} B_{j,k}.
    # So we need:
    # sum_c W_{k,i,c} U_{i,j,c} V_{l,k,c} = 1 if j == l, else 0.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    expected = 1 if j == l else 0
                    calculated = 0
                    for c in range(rank):
                        # U index: i*m + j
                        # V index: l*p + k
                        # W index: k*n + i
                        u_val = U[i*m + j][c]
                        v_val = V[l*p + k][c]
                        w_val = W[k*n + i][c]
                        calculated += u_val * v_val * w_val
                    if calculated != expected:
                        return False
    return True

# EVOLVE-BLOCK-END
