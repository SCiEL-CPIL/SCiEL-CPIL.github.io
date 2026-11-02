"""Matrix multiplication 2x4 by 4x8 using Strassen decomposition."""

def construct():
    n, m, p = 2, 4, 8
    # Total rank from Strassen: 4 subproblems of 2x2*2x2, each 7 mults
    # A is 2x4, split into A_left (2x2) cols 0-1, A_right (2x2) cols 2-3
    # B is 4x8, split into B_top (2x8) rows 0-1, B_bottom (2x8) rows 2-3
    # C = A_left*B_top + A_right*B_bottom
    # Each 2x2 * 2x8 = two 2x2*2x2 Strassen blocks
    # Total: 4 * 7 = 28? No wait...
    # A_left * B_top: B_top is 2x8, split into B_top_left (2x4) and B_top_right (2x4)
    # Each 2x4 = two 2x2 blocks, so 2 Strassen blocks per 2x2*2x4
    # A_left*B_top = 2 Strassen blocks = 14
    # A_right*B_bottom = 2 Strassen blocks = 14
    # Total = 28? But measured was 56...
    # Wait, 2x4 * 2x8: the 4 columns of A map to 4 rows of B.
    # Let me re-examine.
    
    # Actually: A is (2,4), B is (4,8). C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Split j into j=0,1 and j=2,3:
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # A[:,0:2] is 2x2, B[0:2,:] is 2x8
    # 2x2 * 2x8: for each output column k, C[:,k] = A_sub * B_sub[:,k]
    # This is 2x2 * 2 * (4 outputs) = 4 independent 2x2*2x2 problems
    # Each uses 7 mults, so 4*7 = 28 for the first term
    # Similarly 28 for the second term
    # Total: 56. Yes!
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    
    # Strassen 7-multiplication pattern for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Returns list of (U_coeffs, V_coeffs, W_coeffs) for each of 7 mults
    # U_coeffs: (coeff_a, coeff_b, coeff_c, coeff_d)
    # V_coeffs: (coeff_e, coeff_f, coeff_g, coeff_h)
    # W_coeffs: (coeff_C11, coeff_C12, coeff_C21, coeff_C22)
    
    def strassen_terms():
        # M1 = (a+d)(e+h) -> C11+=1, C22+=1
        yield ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1))
        # M2 = (c+d)e -> C21+=1, C22+=-1
        yield ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1))
        # M3 = a(f-h) -> C12+=1, C22+=1
        yield ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1))
        # M4 = d(g-e) -> C11+=1, C21+=1
        yield ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0))
        # M5 = (a+b)h -> C11+=-1, C12+=1
        yield ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0))
        # M6 = (c-a)(e+f) -> C22+=1
        yield ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1))
        # M7 = (b-d)(g+h) -> C11+=1
        yield ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0))
    
    def add_strassen_block(a_rows, b_rows, c_rows):
        """Add 7 Strassen terms for a 2x2*2x2 block.
        a_rows: [row0, row1] for A entries [a, b; c, d] -> a_rows[0]=a, a_rows[1]=b, etc.
                 Actually: a_rows = [idx_a, idx_b, idx_c, idx_d]
        b_rows: [idx_e, idx_f, idx_g, idx_h]
        c_rows: [idx_C11, idx_C12, idx_C21, idx_C22]
        Returns new column index.
        """
        nonlocal col
        for u_c, v_c, w_c in strassen_terms():
            for idx, coeff in zip(a_rows, u_c):
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in zip(b_rows, v_c):
                if coeff != 0:
                    V[idx][col] += coeff
            for idx, coeff in zip(c_rows, w_c):
                if coeff != 0:
                    W[idx][col] += coeff
            col += 1
        return col
    
    # A is 2x4: A[i,j] for i in 0..1, j in 0..3
    # U row for A[i,j] = i*m + j = i*4 + j
    # B is 4x8: B[j,k] for j in 0..3, k in 0..7
    # V row for B[j,k] = j*p + k = j*8 + k
    # C is 2x8: C[i,k] for i in 0..1, k in 0..7
    # W row for C[i,k] = k*n + i (trace convention W[k*n+i])
    
    # Split: j=0,1 (top) and j=2,3 (bottom)
    # For j in {0,1}: A_left = A[:, 0:2], B_top = B[0:2, :]
    # For j in {2,3}: A_right = A[:, 2:4], B_bottom = B[2:4, :]
    
    for j_group in range(2):
        # j_group=0: j=0,1; j_group=1: j=2,3
        j0 = j_group * 2
        j1 = j_group * 2 + 1
        
        # A subblock: 2x2 with entries A[0,j0], A[0,j1], A[1,j0], A[1,j1]
        # U rows: a=A[0,j0]=0*4+j0, b=A[0,j1]=0*4+j1, c=A[1,j0]=1*4+j0, d=A[1,j1]=1*4+j1
        a_rows = [0 * 4 + j0, 0 * 4 + j1, 1 * 4 + j0, 1 * 4 + j1]
        
        # B subblock rows: B[j0, k] and B[j1, k] for k in 0..7
        # For each pair of output columns (k0, k1), B_sub = [[B[j0,k0], B[j0,k1]], [B[j1,k0], B[j1,k1]]]
        # C subblock: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # W rows: C[i,k] -> k*n + i = k*2 + i
        
        for k_group in range(4):
            k0 = k_group * 2
            k1 = k_group * 2 + 1
            
            # B entries: e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
            b_rows = [j0 * 8 + k0, j0 * 8 + k1, j1 * 8 + k0, j1 * 8 + k1]
            
            # C entries: C11=C[0,k0], C12=C[0,k1], C21=C[1,k0], C22=C[1,k1]
            c_rows = [k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1]
            
            add_strassen_block(a_rows, b_rows, c_rows)
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
