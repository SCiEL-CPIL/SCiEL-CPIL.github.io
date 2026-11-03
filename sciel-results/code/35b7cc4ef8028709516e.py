"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 Strassen 2x2 blocks?
    # Actually, 4 blocks of C, each is sum of 2 2x2 products.
    # Each 2x2 product uses 7 mults.
    # Total 4 * 2 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen multipliers for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a+d)(h)
    # M5 = (a+c)(e-g)
    # M6 = (b+d)(g+h)
    # M7 = (a-c)(f-h)
    
    # We need to map these to the global indices.
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # B is 4x8. Rows 0,1,2,3. Cols 0..7.
    # C is 2x8. Rows 0,1. Cols 0..7.
    
    # Block j (j=0..3) of C is C[:, 2j:2j+2].
    # C[:, 2j:2j+2] = A[:, 2j:2j+2] B[2j:2j+2, :] + A[:, 2j+2:2j+4] B[2j+2:2j+4, :]
    # Let A1 = A[:, 2j:2j+2], A2 = A[:, 2j+2:2j+4]
    # Let B1 = B[2j:2j+2, :], B2 = B[2j+2:2j+4, :]
    # Cj = A1 B1 + A2 B2
    
    # We use Strassen for A1 B1 and A2 B2.
    # Total 7 + 7 = 14 mults per block.
    
    col = 0
    for j in range(4):
        # A1 cols: 2j, 2j+1
        # A2 cols: 2j+2, 2j+3
        # B1 rows: 2j, 2j+1
        # B2 rows: 2j+2, 2j+3
        # C cols: 2j, 2j+1
        
        # Strassen for A1 B1
        # M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,1])
        # Indices in A: i=0,1; j=2j, 2j+1
        # Indices in B: i=2j, 2j+1; j=2j, 2j+1 (for B1)
        
        # Helper to add a Strassen term
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_cols, sign=1):
            nonlocal col
            # a_rows, a_cols, b_rows, b_cols are lists of indices
            # We assume 2x2 structure
            a00, a01 = a_rows[0], a_cols[0]
            a10, a11 = a_rows[1], a_cols[1]
            b00, b01 = b_rows[0], b_cols[0]
            b10, b11 = b_rows[1], b_cols[1]
            c00, c01 = c_rows[0], c_cols[0]
            c10, c11 = c_rows[1], c_cols[1]
            
            # M1
            U[a00][col] += sign
            U[a11][col] += sign
            V[b00][col] += sign
            V[b11][col] += sign
            W[c00][col] += 1
            W[c11][col] += 1
            W[c01][col] -= 1
            W[c10][col] -= 1
            col += 1
            
            # M2
            U[a10][col] += sign
            U[a11][col] += sign
            V[b00][col] += sign
            W[c01][col] += 1
            W[c11][col] += 1
            col += 1
            
            # M3
            U[a00][col] += sign
            U[a01][col] -= sign
            V[b01][col] += sign
            V[b11][col] += sign
            W[c00][col] += 1
            W[c10][col] += 1
            col += 1
            
            # M4
            U[a00][col] += sign
            U[a11][col] += sign
            V[b11][col] += sign
            W[c00][col] -= 1
            W[c11][col] -= 1
            W[c01][col] += 1
            W[c10][col] += 1
            col += 1
            
            # M5
            U[a00][col] += sign
            U[a10][col] += sign
            V[b00][col] += sign
            V[b10][col] -= sign
            W[c10][col] += 1
            col += 1
            
            # M6
            U[a01][col] += sign
            U[a11][col] += sign
            V[b10][col] += sign
            V[b11][col] += sign
            W[c01][col] += 1
            col += 1
            
            # M7
            U[a00][col] += sign
            U[a10][col] -= sign
            V[b01][col] += sign
            V[b11][col] -= sign
            W[c00][col] -= 1
            W[c10][col] -= 1
            col += 1

        # A1 B1
        a_rows = [0, 1]
        a_cols = [2*j, 2*j+1]
        b_rows = [2*j, 2*j+1]
        b_cols = [2*j, 2*j+1]
        c_rows = [0, 1]
        c_cols = [2*j, 2*j+1]
        add_strassen(a_rows, a_cols, b_rows, b_cols, c_cols, 1)
        
        # A2 B2
        a_rows = [0, 1]
        a_cols = [2*j+2, 2*j+3]
        b_rows = [2*j+2, 2*j+3]
        b_cols = [2*j, 2*j+1]
        c_rows = [0, 1]
        c_cols = [2*j, 2*j+1]
        add_strassen(a_rows, a_cols, b_rows, b_cols, c_cols, 1)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
