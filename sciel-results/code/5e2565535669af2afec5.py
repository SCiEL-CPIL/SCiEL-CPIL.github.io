import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Initialize U, V, W as lists of lists
    # We will build them column by column (term by term)
    terms = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        # u_indices: dict {index: value}
        # v_indices: dict {index: value}
        # w_indices: dict {index: value}
        term_u = [0.0] * (n * m)
        term_v = [0.0] * (m * p)
        term_w = [0.0] * (p * n)
        
        for idx, val in u_indices.items():
            term_u[idx] = val
        for idx, val in v_indices.items():
            term_v[idx] = val
        for idx, val in w_indices.items():
            term_w[idx] = val
            
        terms.append((term_u, term_v, term_w))

    # Strassen Block Generator
    def add_strassen_block(i0, j0, k0):
        # i0, j0, k0 are start indices for the 2x2x2 block
        # Local indices 0,1 map to global i0, i0+1 etc.
        
        # P1
        u = {}
        v = {}
        w = {}
        # u: A00, A11
        u[i0 * m + j0] = 1.0
        u[(i0+1) * m + (j0+1)] = 1.0
        # v: B00, B11
        v[j0 * p + k0] = 1.0
        v[(j0+1) * p + (k0+1)] = 1.0
        # w: C00 (+), C11 (+)
        # w[k, i] -> index k*n + i
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + (i0+1)] = 1.0
        add_term(u, v, w)

        # P2
        u = {}
        v = {}
        w = {}
        # u: A10, A11
        u[(i0+1) * m + j0] = 1.0
        u[(i0+1) * m + (j0+1)] = 1.0
        # v: B00
        v[j0 * p + k0] = 1.0
        # w: C10 (+) -> i=1, k=0
        w[k0 * n + (i0+1)] = 1.0
        add_term(u, v, w)

        # P3
        u = {}
        v = {}
        w = {}
        # u: A00
        u[i0 * m + j0] = 1.0
        # v: B01, B11
        v[j0 * p + (k0+1)] = 1.0
        v[(j0+1) * p + (k0+1)] = -1.0
        # w: C01 (+) -> i=0, k=1
        w[(k0+1) * n + i0] = 1.0
        add_term(u, v, w)

        # P4
        u = {}
        v = {}
        w = {}
        # u: A11
        u[(i0+1) * m + (j0+1)] = 1.0
        # v: B10, B00
        v[(j0+1) * p + k0] = 1.0
        v[j0 * p + k0] = -1.0
        # w: C00 (+), C10 (+)
        w[k0 * n + i0] = 1.0
        w[k0 * n + (i0+1)] = 1.0
        add_term(u, v, w)

        # P5
        u = {}
        v = {}
        w = {}
        # u: A00, A01
        u[i0 * m + j0] = 1.0
        u[i0 * m + (j0+1)] = 1.0
        # v: B11
        v[(j0+1) * p + (k0+1)] = 1.0
        # w: C00 (-), C01 (+)
        w[k0 * n + i0] = -1.0
        w[(k0+1) * n + i0] = 1.0
        add_term(u, v, w)

        # P6
        u = {}
        v = {}
        w = {}
        # u: A10, A00
        u[(i0+1) * m + j0] = 1.0
        u[i0 * m + j0] = -1.0
        # v: B00, B01
        v[j0 * p + k0] = 1.0
        v[j0 * p + (k0+1)] = 1.0
        # w: C11 (+)
        w[(k0+1) * n + (i0+1)] = 1.0
        add_term(u, v, w)

        # P7
        u = {}
        v = {}
        w = {}
        # u: A01, A11
        u[i0 * m + (j0+1)] = 1.0
        u[(i0+1) * m + (j0+1)] = -1.0
        # v: B10, B11
        v[(j0+1) * p + k0] = 1.0
        v[(j0+1) * p + (k0+1)] = 1.0
        # w: C00 (+)
        w[k0 * n + i0] = 1.0
        add_term(u, v, w)

    # Rank 1 Block Generator for 2x1x2 (A 2x1, B 1x2)
    # A rows i0, i0+1. A col j0. B row j0. B cols k0, k0+1.
    def add_rank1_212_block(i0, j0, k0):
        u = {}
        v = {}
        w = {}
        # u: A_i0_j0, A_i0+1_j0
        u[i0 * m + j0] = 1.0
        u[(i0+1) * m + j0] = 1.0
        # v: B_j0_k0, B_j0_k0+1
        v[j0 * p + k0] = 1.0
        v[j0 * p + (k0+1)] = 1.0
        # w: C_i0_k0, C_i0_k0+1, C_i0+1_k0, C_i0+1_k0+1
        # w[k, i]
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + i0] = 1.0
        w[k0 * n + (i0+1)] = 1.0
        w[(k0+1) * n + (i0+1)] = 1.0
        add_term(u, v, w)

    # Rank 1 Block Generator for 1x2x2 (A 1x2, B 2x2)
    # A row i0. A cols j0, j0+1. B rows j0, j0+1. B cols k0, k0+1.
    def add_rank1_122_block(i0, j0, k0):
        u = {}
        v = {}
        w = {}
        # u: A_i0_j0, A_i0_j0+1
        u[i0 * m + j0] = 1.0
        u[i0 * m + (j0+1)] = 1.0
        # v: B_j0_k0, B_j0_k0+1, B_j0+1_k0, B_j0+1_k0+1
        v[j0 * p + k0] = 1.0
        v[j0 * p + (k0+1)] = 1.0
        v[(j0+1) * p + k0] = 1.0
        v[(j0+1) * p + (k0+1)] = 1.0
        # w: C_i0_k0, C_i0_k0+1
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + i0] = 1.0
        add_term(u, v, w)

    # Rank 1 Block Generator for 1x1x2 (A 1x1, B 1x2)
    # A row i0. A col j0. B row j0. B cols k0, k0+1.
    def add_rank1_112_block(i0, j0, k0):
        u = {}
        v = {}
        w = {}
        # u: A_i0_j0
        u[i0 * m + j0] = 1.0
        # v: B_j0_k0, B_j0_k0+1
        v[j0 * p + k0] = 1.0
        v[j0 * p + (k0+1)] = 1.0
        # w: C_i0_k0, C_i0_k0+1
        w[k0 * n + i0] = 1.0
        w[(k0+1) * n + i0] = 1.0
        add_term(u, v, w)

    # Apply blocks
    # 1. A00 B00 -> C00 (2x2x2)
    add_strassen_block(0, 0, 0)
    
    # 2. A01 B10 -> C00 (2x2x2)
    # A01 is rows 0,1 cols 2,3. B10 is rows 2,3 cols 0,1.
    # Output C00 is rows 0,1 cols 0,1.
    add_strassen_block(0, 2, 0)
    
    # 3. A01 B11 -> C01 (2x2x2)
    # A01 rows 0,1 cols 2,3. B11 rows 2,3 cols 2,3.
    # Output C01 is rows 0,1 cols 2,3.
    add_strassen_block(0, 2, 2)
    
    # 4. A01 B12 -> C02 (2x2x2)
    # A01 rows 0,1 cols 2,3. B12 rows 2,3 cols 4,5.
    # Output C02 is rows 0,1 cols 4,5.
    add_strassen_block(0, 2, 4)
    
    # 5. A02 B20 -> C00 (2x1x2)
    # A02 rows 0,1 col 4. B20 row 4 cols 0,1.
    add_rank1_212_block(0, 4, 0)
    
    # 6. A02 B21 -> C01 (2x1x2)
    # A02 rows 0,1 col 4. B21 row 4 cols 2,3.
    add_rank1_212_block(0, 4, 2)
    
    # 7. A02 B22 -> C02 (2x1x2)
    # A02 rows 0,1 col 4. B22 row 4 cols 4,5.
    add_rank1_212_block(0, 4, 4)
    
    # 8. A10 B00 -> C10 (1x2x2)
    # A10 row 2 cols 0,1. B00 rows 0,1 cols 0,1.
    add_rank1_122_block(2, 0, 0)
    
    # 9. A10 B01 -> C11 (1x2x2)
    # A10 row 2 cols 0,1. B01 rows 0,1 cols 2,3.
    add_rank1_122_block(2, 0, 2)
    
    # 10. A10 B02 -> C12 (1x2x2)
    # A10 row 2 cols 0,1. B02 rows 0,1 cols 4,5.
    add_rank1_122_block(2, 0, 4)
    
    # 11. A11 B10 -> C10 (1x2x2)
    # A11 row 2 cols 2,3. B10 rows 2,3 cols 0,1.
    add_rank1_122_block(2, 2, 0)
    
    # 12. A11 B11 -> C11 (1x2x2)
    # A11 row 2 cols 2,3. B11 rows 2,3 cols 2,3.
    add_rank1_122_block(2, 2, 2)
    
    # 13. A11 B12 -> C12 (1x2x2)
    # A11 row 2 cols 2,3. B12 rows 2,3 cols 4,5.
    add_rank1_122_block(2, 2, 4)
    
    # 14. A12 B20 -> C10 (1x1x2)
    # A12 row 2 col 4. B20 row 4 cols 0,1.
    add_rank1_112_block(2, 4, 0)
    
    # 15. A12 B21 -> C11 (1x1x2)
    # A12 row 2 col 4. B21 row 4 cols 2,3.
    add_rank1_112_block(2, 4, 2)
    
    # 16. A12 B22 -> C12 (1x1x2)
    # A12 row 2 col 4. B22 row 4 cols 4,5.
    add_rank1_112_block(2, 4, 4)
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u, v, w = terms[r]
        for i in range(n * m):
            U[i][r] = u[i]
        for j in range(m * p):
            V[j][r] = v[j]
        for k in range(p * n):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}
