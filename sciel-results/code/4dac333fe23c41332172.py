"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We'll build U, V, W as lists of lists, then determine rank.
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (27, rank)
    # W: (p*n, rank) = (18, rank)
    
    U_cols = []  # Each element is a list of length n*m
    V_cols = []  # Each element is a list of length m*p
    W_cols = []  # Each element is a list of length p*n
    
    def add_term(U_entry, V_entry, W_entry):
        """Add a rank-1 term. U_entry, V_entry, W_entry are dicts mapping index -> value."""
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Part 1: A1 * B1 where A1 is 2x2 (cols 0,1 of A), B1 is 2x9 (rows 0,1 of B)
    # Split 9 columns into 4 pairs + 1 single
    
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        # Strassen for 2x2 x 2x2
        # M1 = (a00+a11)*(b00+b11)
        add_term(
            {0*m+0: 1, 1*m+1: 1},  # A[0,0] + A[1,1]
            {0*p+k0: 1, 1*p+k1: 1},  # B[0,k0] + B[1,k1]
            {k0*n+0: 1, k1*n+1: 1}  # C[0,k0] + C[1,k1]
        )
        # M2 = (a10+a11)*b00
        add_term(
            {1*m+0: 1, 1*m+1: 1},  # A[1,0] + A[1,1]
            {0*p+k0: 1},  # B[0,k0]
            {k0*n+1: 1, k1*n+1: -1}  # C[1,k0] - C[1,k1]
        )
        # M3 = a00*(b01-b11)
        add_term(
            {0*m+0: 1},  # A[0,0]
            {0*p+k1: 1, 1*p+k1: -1},  # B[0,k1] - B[1,k1]
            {k1*n+0: 1, k1*n+1: 1}  # C[0,k1] + C[1,k1]
        )
        # M4 = a11*(b10-b00)
        add_term(
            {1*m+1: 1},  # A[1,1]
            {1*p+k0: 1, 0*p+k0: -1},  # B[1,k0] - B[0,k0]
            {k0*n+0: 1, k0*n+1: 1}  # C[0,k0] + C[1,k0]
        )
        # M5 = (a00+a01)*b11
        add_term(
            {0*m+0: 1, 0*m+1: 1},  # A[0,0] + A[0,1]
            {1*p+k1: 1},  # B[1,k1]
            {k0*n+0: -1, k1*n+0: 1}  # -C[0,k0] + C[0,k1]
        )
        # M6 = (a10-a00)*(b00+b01)
        add_term(
            {1*m+0: 1, 0*m+0: -1},  # A[1,0] - A[0,0]
            {0*p+k0: 1, 0*p+k1: 1},  # B[0,k0] + B[0,k1]
            {k1*n+1: 1}  # C[1,k1]
        )
        # M7 = (a01-a11)*(b10+b11)
        add_term(
            {0*m+1: 1, 1*m+1: -1},  # A[0,1] - A[1,1]
            {1*p+k0: 1, 1*p+k1: 1},  # B[1,k0] + B[1,k1]
            {k0*n+0: 1}  # C[0,k0]
        )
    
    # Single column k=8: schoolbook 2x2 x 2x1
    k = 8
    # A[0,0]*B[0,k] -> C[0,k]
    add_term({0*m+0: 1}, {0*p+k: 1}, {k*n+0: 1})
    # A[0,1]*B[1,k] -> C[0,k]
    add_term({0*m+1: 1}, {1*p+k: 1}, {k*n+0: 1})
    # A[1,0]*B[0,k] -> C[1,k]
    add_term({1*m+0: 1}, {0*p+k: 1}, {k*n+1: 1})
    # A[1,1]*B[1,k] -> C[1,k]
    add_term({1*m+1: 1}, {1*p+k: 1}, {k*n+1: 1})
    
    # Part 2: A2 * B2 where A2 is 2x1 (col 2 of A), B2 is 1x9 (row 2 of B)
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},  # A[i,2]
                {2*p+k: 1},  # B[2,k]
                {k*n+i: 1}   # C[i,k]
            )
    
    rank = len(U_cols)
    
    # Transpose to get (n*m, rank), (m*p, rank), (p*n, rank)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
