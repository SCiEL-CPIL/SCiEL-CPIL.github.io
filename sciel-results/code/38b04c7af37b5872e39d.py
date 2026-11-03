"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in two parts:
    # Part 1: M_{2,2,6} using A[:, 0:2] and B[0:2, :]
    # Part 2: M_{2,3,6} using A[:, 2:5] and B[2:5, :]
    
    # We accumulate the columns of U, V, W
    # Start with empty lists of rows
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term to the decomposition
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(j,k): val}
    # w_coeffs: dict {(k,i): val}
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # Determine the max indices to define row sizes? 
        # No, we know n, m, p.
        # We need to append a column to U, V, W.
        # U is n*m x rank
        # V is m*p x rank
        # W is p*n x rank
        
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            u_col[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            v_col[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            w_col[k * n + i] = val
            
        U_rows.append(u_col)
        V_rows.append(v_col)
        W_rows.append(w_col)

    # Helper: Strassen 2x2x2
    # Maps local i, j, k in {0,1} to global gi, gj, gk
    def add_strassen(gi_offset, gj_offset, gk_offset):
        # Local i, j, k -> Global gi, gj, gk
        # gi = gi_offset + i
        # gj = gj_offset + j
        # gk = gk_offset + k
        
        # Term 1
        add_term(
            {(gi_offset+0, gj_offset+0): 1, (gi_offset+1, gj_offset+1): 1},
            {(gj_offset+0, gk_offset+0): 1, (gj_offset+1, gk_offset+1): 1},
            {(gk_offset+0, gi_offset+0): 1, (gk_offset+1, gi_offset+1): 1}
        )
        # Term 2
        add_term(
            {(gi_offset+1, gj_offset+0): 1, (gi_offset+1, gj_offset+1): 1},
            {(gj_offset+0, gk_offset+0): 1},
            {(gk_offset+0, gi_offset+1): 1, (gk_offset+1, gi_offset+1): -1}
        )
        # Term 3
        add_term(
            {(gi_offset+0, gj_offset+0): 1},
            {(gj_offset+1, gk_offset+0): 1, (gj_offset+1, gk_offset+1): -1},
            {(gk_offset+0, gi_offset+1): 1, (gk_offset+1, gi_offset+1): 1}
        )
        # Term 4
        add_term(
            {(gi_offset+1, gj_offset+1): 1},
            {(gj_offset+0, gk_offset+1): 1, (gj_offset+0, gk_offset+0): -1},
            {(gk_offset+0, gi_offset+0): 1, (gk_offset+1, gi_offset+1): 1}
        )
        # Term 5
        add_term(
            {(gi_offset+0, gj_offset+0): 1, (gi_offset+0, gj_offset+1): 1},
            {(gj_offset+1, gk_offset+1): 1},
            {(gk_offset+0, gi_offset+0): -1, (gk_offset+1, gi_offset+1): 1}
        )
        # Term 6
        add_term(
            {(gi_offset+1, gj_offset+0): 1, (gi_offset+0, gj_offset+0): -1},
            {(gj_offset+0, gk_offset+0): 1, (gj_offset+0, gk_offset+1): 1},
            {(gk_offset+1, gi_offset+1): 1}
        )
        # Term 7
        add_term(
            {(gi_offset+0, gj_offset+1): 1, (gi_offset+1, gj_offset+1): -1},
            {(gj_offset+1, gk_offset+0): 1, (gj_offset+1, gk_offset+1): 1},
            {(gk_offset+0, gi_offset+0): 1}
        )

    # Helper: Schoolbook n x m x p (for small dims like 1)
    def add_schoolbook(gi_offset, gj_offset, gk_offset, ni, mj, pk):
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    add_term(
                        {(gi_offset+i, gj_offset+j): 1},
                        {(gj_offset+j, gk_offset+k): 1},
                        {(gk_offset+k, gi_offset+i): 1}
                    )

    # --- Part 1: M_{2,2,6} ---
    # A[:, 0:2] (2x2), B[0:2, :] (2x6)
    # Split output 6 into three 2s: k in {0,1}, {2,3}, {4,5}
    for s in range(3):
        gk_offset = s * 2
        add_strassen(0, 0, gk_offset)

    # --- Part 2: M_{2,3,6} ---
    # A[:, 2:5] (2x3), B[2:5, :] (3x6)
    # Split output 6 into three 2s: k in {0,1}, {2,3}, {4,5}
    for s in range(3):
        gk_offset = s * 2
        
        # Sub-part 2a: M_{2,2,2} using j in {0,1} of local block (Global j 2,3)
        # Local j 0 -> Global j 2
        # Local j 1 -> Global j 3
        add_strassen(0, 2, gk_offset)
        
        # Sub-part 2b: M_{2,1,2} using j=2 of local block (Global j 4)
        # Schoolbook 2x1x2
        add_schoolbook(0, 4, gk_offset, 2, 1, 2)

    # Convert lists of columns to matrices
    # U is n*m rows, rank columns
    # We have lists of columns. We need to transpose them or build row-wise.
    # Current U_rows is a list of length 'rank', each element is a list of length 'n*m'.
    # We need U[i][r] = U_rows[r][i]
    
    rank = len(U_rows)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_rows[r][i]
        for j in range(m * p):
            V[j][r] = V_rows[r][j]
        for k in range(p * n):
            W[k][r] = W_rows[r][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
