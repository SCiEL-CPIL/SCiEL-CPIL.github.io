"""Matrix multiplication 2x2 x 2x13 - tensor decomposition search."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 6 pairs of columns (2x2x2) + 1 single column (2x2x1)
    # Rank of 2x2x2 (Strassen) = 7
    # Rank of 2x2x1 (Schoolbook) = 4
    rank = 6 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen term for a pair of columns
    def add_strassen_term(pair_idx):
        nonlocal col_idx
        # Columns 2*pair_idx and 2*pair_idx + 1
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # V indices for e_k0, f_k0, e_k1, f_k1
        # V row index: j*p + k. Here j=0 is top row of B, j=1 is bottom row.
        # B is 2x13.
        # V[0*p + k] = e_k (top row of B, col k)
        # V[1*p + k] = f_k (bottom row of B, col k)
        
        # Strassen M1: (a+d)(e+h)
        # a=U[0], d=U[3]
        # e=V[p+k0], h=V[p+k1]
        U[0][col_idx] = 1
        U[3][col_idx] = 1
        V[p + k0][col_idx] = 1
        V[p + k1][col_idx] = 1
        # W contributions:
        # C_00 (k0, 0) gets +1
        # C_11 (k1, 1) gets +1
        W[k0 * n + 0][col_idx] = 1
        W[k1 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Strassen M2: (c+d)e
        # c=U[2], d=U[3]
        # e=V[p+k0]
        U[2][col_idx] = 1
        U[3][col_idx] = 1
        V[p + k0][col_idx] = 1
        # W contributions:
        # C_10 (k0, 1) gets +1
        W[k0 * n + 1][col_idx] = 1
        # C_11 (k1, 1) gets -1
        W[k1 * n + 1][col_idx] = -1
        col_idx += 1
        
        # Strassen M3: a(f-h)
        # a=U[0]
        # f=V[p+k1], h=V[p+k1] -> Wait.
        # B = [[e0, e1], [f0, f1]]
        # e=e0, f=e1, g=f0, h=f1
        # M3 = a(f-h) = a(e1 - f1)
        # V indices: e1 is p+k1, f1 is p+k1? No.
        # V[p+k] is top row (e). V[p*1+k] is bottom row (f).
        # So e1 = V[p+k1], f1 = V[p+k1] is WRONG.
        # f1 = V[1*p + k1] = V[p+k1].
        # e1 = V[0*p + k1] = V[k1].
        # So M3 = a(V[k1] - V[p+k1])
        U[0][col_idx] = 1
        V[k1][col_idx] = 1
        V[p + k1][col_idx] = -1
        # W contributions:
        # C_01 (k1, 0) gets +1
        W[k1 * n + 0][col_idx] = 1
        # C_11 (k1, 1) gets +1
        W[k1 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Strassen M4: d(g-e)
        # d=U[3]
        # g=f0=V[p+k0], e=e0=V[k0]
        # M4 = d(V[p+k0] - V[k0])
        U[3][col_idx] = 1
        V[p + k0][col_idx] = 1
        V[k0][col_idx] = -1
        # W contributions:
        # C_00 (k0, 0) gets -1
        W[k0 * n + 0][col_idx] = -1
        # C_10 (k0, 1) gets +1
        W[k0 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Strassen M5: (a+b)h
        # a=U[0], b=U[1]
        # h=f1=V[p+k1]
        U[0][col_idx] = 1
        U[1][col_idx] = 1
        V[p + k1][col_idx] = 1
        # W contributions:
        # C_00 (k0, 0) gets +1
        W[k0 * n + 0][col_idx] = 1
        # C_01 (k1, 0) gets +1
        W[k1 * n + 0][col_idx] = 1
        col_idx += 1
        
        # Strassen M6: (c-a)(e+f)
        # c=U[2], a=U[0]
        # e=e0=V[k0], f=e1=V[k1]
        # M6 = (c-a)(V[k0] + V[k1])
        U[2][col_idx] = 1
        U[0][col_idx] = -1
        V[k0][col_idx] = 1
        V[k1][col_idx] = 1
        # W contributions:
        # C_11 (k1, 1) gets +1
        W[k1 * n + 1][col_idx] = 1
        col_idx += 1
        
        # Strassen M7: (b-d)(g+h)
        # b=U[1], d=U[3]
        # g=f0=V[p+k0], h=f1=V[p+k1]
        # M7 = (b-d)(V[p+k0] + V[p+k1])
        U[1][col_idx] = 1
        U[3][col_idx] = -1
        V[p + k0][col_idx] = 1
        V[p + k1][col_idx] = 1
        # W contributions:
        # C_00 (k0, 0) gets -1
        W[k0 * n + 0][col_idx] = -1
        col_idx += 1

    # Process 6 pairs
    for pair_idx in range(6):
        add_strassen_term(pair_idx)
        
    # Process single column k=12
    k_single = 12
    # P1: a * e
    U[0][col_idx] = 1
    V[k_single][col_idx] = 1
    W[k_single * n + 0][col_idx] = 1
    col_idx += 1
    
    # P2: b * f
    U[1][col_idx] = 1
    V[p + k_single][col_idx] = 1
    W[k_single * n + 0][col_idx] = 1
    col_idx += 1
    
    # P3: c * e
    U[2][col_idx] = 1
    V[k_single][col_idx] = 1
    W[k_single * n + 1][col_idx] = 1
    col_idx += 1
    
    # P4: d * f
    U[3][col_idx] = 1
    V[p + k_single][col_idx] = 1
    W[k_single * n + 1][col_idx] = 1
    col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
