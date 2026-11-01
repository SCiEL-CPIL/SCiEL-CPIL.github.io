"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Rank 50 via Strassen block decomposition
    rank = 50
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # i0: row block start in A (0 or 1, but here always 0 since n=2)
    # j0: col block start in A (0 or 2)
    # k0: col block start in B (0, 2, 4)
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        # A indices: A[i0, j0], A[i0, j0+1], A[i0+1, j0], A[i0+1, j0+1]
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices: B[j0, k0], B[j0, k0+1], B[j0+1, k0], B[j0+1, k0+1]
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices: W[k*n + i]
        # C[i0, k0], C[i0, k0+1], C[i0+1, k0], C[i0+1, k0+1]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        # Contributes: C00 += 1, C11 += 1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        # Contributes: C10 += 1, C11 -= 1
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        # Contributes: C01 += 1, C11 -= 1
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M4 = d(g-e)
        # Contributes: C00 += 1, C10 += 1
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        # Contributes: C00 -= 1, C01 += 1
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # Contributes: C11 += 1
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # Contributes: C00 += 1
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    # i0: row block start in A
    # j0: col block start in A (2-wide)
    # k0: single col in B
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Process blocks
    # Split inner dimension m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # For each block, split output p=7 into three pairs (0,1),(2,3),(4,5) and one singleton k=6
    for j_start in [0, 2]:
        # Three Strassen blocks for k pairs
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # One schoolbook block for k=6
        add_schoolbook_221_block(0, j_start, 6)
    
    # col should be exactly 50
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
