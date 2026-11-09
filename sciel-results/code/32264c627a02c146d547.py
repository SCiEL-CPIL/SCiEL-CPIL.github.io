"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    
    The decomposition splits the multiplication into:
    1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Rank 7, Strassen)
    2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Rank 7, Strassen)
    3. A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Rank 4, Schoolbook)
    4. A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5] (Rank 10, Schoolbook)
    
    Total Rank = 7 + 7 + 4 + 10 = 28.
    """
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_term_multi(u_vals, v_vals, w_vals):
        terms.append((u_vals, v_vals, w_vals))

    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols):
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = b_cols
        
        # M1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        add_term_multi(
            {i1*m+j1: 1, i2*m+j2: 1},
            {j1*p+k1: 1, j2*p+k2: 1},
            {k1*n+i1: 1, k2*n+i2: 1}
        )
        
        # M2: (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        add_term_multi(
            {i2*m+j1: 1, i2*m+j2: 1},
            {j1*p+k1: 1},
            {k1*n+i2: 1, k2*n+i2: -1}
        )
        
        # M3: A[i1,j1] * (B[j1,k2] - B[j2,k2])
        add_term_multi(
            {i1*m+j1: 1},
            {j1*p+k2: 1, j2*p+k2: -1},
            {k2*n+i1: 1, k2*n+i2: 1}
        )
        
        # M4: A[i2,j2] * (B[j2,k1] - B[j1,k1])
        add_term_multi(
            {i2*m+j2: 1},
            {j2*p+k1: 1, j1*p+k1: -1},
            {k1*n+i1: 1, k1*n+i2: 1}
        )
        
        # M5: (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        add_term_multi(
            {i1*m+j1: 1, i1*m+j2: 1},
            {j2*p+k2: 1},
            {k1*n+i1: -1, k2*n+i1: 1}
        )
        
        # M6: (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        add_term_multi(
            {i2*m+j1: 1, i1*m+j1: -1},
            {j1*p+k1: 1, j1*p+k2: 1},
            {k2*n+i2: 1}
        )
        
        # M7: (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        add_term_multi(
            {i1*m+j2: 1, i2*m+j2: -1},
            {j2*p+k1: 1, j2*p+k2: 1},
            {k1*n+i1: 1}
        )

    # Part 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [0, 1])
    
    # Part 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [2, 3])
    
    # Part 3: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    # C[i, 4] = A[i, 0]B[0, 4] + A[i, 1]B[1, 4]
    add_term_multi({0*m+0: 1}, {0*p+4: 1}, {4*n+0: 1})
    add_term_multi({1*m+0: 1}, {0*p+4: 1}, {4*n+1: 1})
    add_term_multi({0*m+1: 1}, {1*p+4: 1}, {4*n+0: 1})
    add_term_multi({1*m+1: 1}, {1*p+4: 1}, {4*n+1: 1})
    
    # Part 4: A[0:2, 2:3] * B[2:3, 0:5] (2x1x5)
    for i in range(2):
        for k in range(5):
            add_term_multi({i*m+2: 1}, {2*p+k: 1}, {k*n+i: 1})

    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_vals, v_vals, w_vals) in enumerate(terms):
        for idx, val in u_vals.items():
            U[idx][c] = val
        for idx, val in v_vals.items():
            V[idx][c] = val
        for idx, val in w_vals.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
