"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # List to hold the vectors for U, V, W
    # Each entry is a list of length 1 (for now, we will expand to rank)
    # Actually, it's easier to build rows.
    # U has n*m rows, V has m*p rows, W has p*n rows.
    # We will build columns one by one.
    
    # We don't know the rank in advance if we append, but we can estimate.
    # Let's just append.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_indices, v_indices, w_indices):
        # u_indices: list of (index, coeff)
        # v_indices: list of (index, coeff)
        # w_indices: list of (index, coeff)
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, coeff in u_indices:
            u_vec[idx] += coeff
        for idx, coeff in v_indices:
            v_vec[idx] += coeff
        for idx, coeff in w_indices:
            w_vec[idx] += coeff
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen blocks
    # Block (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        # Indices
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # M1 = (A11 + A22)(B11 + B22)
        add_term(
            [(i0*m + j0, 1), (i1*m + j1, 1)],
            [(j0*p + k0, 1), (j1*p + k1, 1)],
            [(k0*n + i0, 1), (k0*n + i1, -1), (k1*n + i0, -1), (k1*n + i1, 1)]
        )
        
        # M2 = (A21 + A22)B11
        add_term(
            [(i1*m + j0, 1), (i1*m + j1, 1)],
            [(j0*p + k0, 1)],
            [(k0*n + i1, 1), (k1*n + i1, -1)] # C21 += M2, C22 -= M2
        )
        
        # M3 = A11 B12
        add_term(
            [(i0*m + j0, 1)],
            [(j0*p + k1, 1)],
            [(k1*n + i0, 1), (k1*n + i1, 1)] # C12 += M3, C22 += M3
        )
        
        # M4 = A22 B21
        add_term(
            [(i1*m + j1, 1)],
            [(j1*p + k0, 1)],
            [(k0*n + i0, 1), (k0*n + i1, 1)] # C11 += M4, C21 += M4
        )
        
        # M5 = (A12 - A22)(B21 + B22)
        add_term(
            [(i0*m + j1, 1), (i1*m + j1, -1)],
            [(j1*p + k0, 1), (j1*p + k1, 1)],
            [(k0*n + i0, -1), (k1*n + i0, 1)] # C11 -= M5, C12 += M5
        )
        
        # M6 = (A11 - A12)(B11 + B12)
        add_term(
            [(i0*m + j0, 1), (i0*m + j1, -1)],
            [(j0*p + k0, 1), (j0*p + k1, 1)],
            [(k1*n + i1, 1)] # C22 += M6
        )
        
        # M7 = (A21 - A11)(B11 + B21)
        add_term(
            [(i1*m + j0, 1), (i0*m + j0, -1)],
            [(j0*p + k0, 1), (j1*p + k0, 1)],
            [(k0*n + i0, -1)] # C11 -= M7
        )
        
    # Define blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    covered = set()
    for b in blocks:
        i0, j0, k0 = b
        add_strassen_block(i0, j0, k0)
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    covered.add((i, j, k))
                    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term(
                        [(i*m + j, 1)],
                        [(j*p + k, 1)],
                        [(k*n + i, 1)]
                    )
                    
    # Convert columns to rows
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_cols[r][i]
        for j in range(m*p):
            V[j][r] = V_cols[r][j]
        for k in range(p*n):
            W[k][r] = W_cols[r][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
