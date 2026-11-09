"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Strategy: split inner dim m=4 into two 2s, output p=5 into 2+2+1
    # Use Strassen (rank 7) for each 2x2x2 sub-block
    
    # U rows indexed by i*m+j where i in [0,2), j in [0,4)
    # V rows indexed by j*p+k where j in [0,4), k in [0,5)
    # W rows indexed by k*n+i where k in [0,5), i in [0,2)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term. u_vec, v_vec, w_vec are lists of length n*m, m*p, p*n."""
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)
    
    def make_u(*entries):
        """Create U vector with given nonzero entries at given positions."""
        vec = [0] * (n * m)
        for idx, val in entries:
            vec[idx] = val
        return vec
    
    def make_v(*entries):
        vec = [0] * (m * p)
        for idx, val in entries:
            vec[idx] = val
        return vec
    
    def make_w(*entries):
        vec = [0] * (p * n)
        for idx, val in entries:
            vec[idx] = val
        return vec
    
    # A is 2x4: A[i][j] for i in 0..1, j in 0..3
    # Split A into A1 = A[:, 0:2] and A2 = A[:, 2:4]
    # B is 4x5: B[j][k] for j in 0..3, k in 0..4
    # Split B into B1 = B[0:2, :] and B2 = B[2:4, :]
    
    # C = A1 @ B1 + A2 @ B2
    # C is 2x5. Split C into C[:, 0:2], C[:, 2:4], C[:, 4]
    
    # For C[:, 0:2] = A1 @ B1[:, 0:2] + A2 @ B2[:, 0:2]
    # This is (2x2)(2x2) + (2x2)(2x2)
    # Use Strassen for each, 7+7=14 terms
    
    # For C[:, 2:4] = A1 @ B1[:, 2:4] + A2 @ B2[:, 2:4]
    # Similarly 14 terms
    
    # For C[:, 4] = A1 @ B1[:, 4] + A2 @ B2[:, 4]
    # (2x2)(2x1) + (2x2)(2x1) = 4+4 = 8 terms (schoolbook)
    
    # Total: 14 + 14 + 8 = 36
    
    # Strassen coefficients for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3-M1+M5+M6
    # C10 = M2-M4+M1-M7
    # C11 = -M3+M2-M6+M1
    
    def strassen_terms(a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_col_offset):
        """
        Add 7 Strassen terms for (2x2)(2x2) -> 2x2.
        a_row_offset: row offset in A (i dimension)
        a_col_offset: col offset in A (j dimension, first of the 2)
        b_row_offset: row offset in B (j dimension)
        b_col_offset: col offset in B (k dimension)
        w_col_offset: col offset in W (k dimension for output)
        
        A indices: i in [a_row_offset, a_row_offset+2), j in [a_col_offset, a_col_offset+2)
        B indices: j in [b_row_offset, b_row_offset+2), k in [b_col_offset, b_col_offset+2)
        C indices: i in [0,2), k in [w_col_offset, w_col_offset+2)
        """
        # a00 = A[a_row_offset][a_col_offset] -> U index: a_row_offset*m + a_col_offset
        # a01 = A[a_row_offset][a_col_offset+1]
        # a10 = A[a_row_offset+1][a_col_offset]
        # a11 = A[a_row_offset+1][a_col_offset+1]
        
        # b00 = B[b_row_offset][b_col_offset] -> V index: b_row_offset*p + b_col_offset
        # b01 = B[b_row_offset][b_col_offset+1]
        # b10 = B[b_row_offset+1][b_col_offset]
        # b11 = B[b_row_offset+1][b_col_offset+1]
        
        # W index for C[i][k]: k*n + i
        
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + b_col_offset + 1
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w00 = w_col_offset * n + 0  # C[0][w_col_offset]
        w01 = w_col_offset * n + 1  # C[1][w_col_offset]
        w10 = (w_col_offset + 1) * n + 0  # C[0][w_col_offset+1]
        w11 = (w_col_offset + 1) * n + 1  # C[1][w_col_offset+1]
        
        # M1 = (a00+a11)(b00+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a11] = 1
        v[b00] = 1; v[b11] = 1
        # C00 += M1, C11 += M1
        w[w00] = 1; w[w11] = 1
        add_term(u, v, w)
        
        # M2 = (a10+a11)b00
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a11] = 1
        v[b00] = 1
        # C10 += M2, C11 += M2
        w[w10] = 1; w[w11] = 1
        add_term(u, v, w)
        
        # M3 = a00(b01-b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1
        v[b01] = 1; v[b11] = -1
        # C01 += M3, C11 -= M3
        w[w01] = 1; w[w11] = -1
        add_term(u, v, w)
        
        # M4 = a11(b10-b00)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a11] = 1
        v[b10] = 1; v[b00] = -1
        # C00 += M4, C10 -= M4
        w[w00] = 1; w[w10] = -1
        add_term(u, v, w)
        
        # M5 = (a00+a01)b11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a00] = 1; u[a01] = 1
        v[b11] = 1
        # C00 -= M5, C01 += M5
        w[w00] = -1; w[w01] = 1
        add_term(u, v, w)
        
        # M6 = (a10-a00)(b00+b01)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a10] = 1; u[a00] = -1
        v[b00] = 1; v[b01] = 1
        # C01 += M6, C11 -= M6
        w[w01] = 1; w[w11] = -1
        add_term(u, v, w)
        
        # M7 = (a01-a11)(b10+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a01] = 1; u[a11] = -1
        v[b10] = 1; v[b11] = 1
        # C00 += M7, C10 -= M7
        w[w00] = 1; w[w10] = -1
        add_term(u, v, w)
    
    # C[:, 0:2] = A1 @ B1[:, 0:2] + A2 @ B2[:, 0:2]
    # A1: rows 0-1, cols 0-1 of A. a_row_offset=0, a_col_offset=0
    # B1[:, 0:2]: rows 0-1 of B, cols 0-1. b_row_offset=0, b_col_offset=0
    # A2: rows 0-1, cols 2-3 of A. a_row_offset=0, a_col_offset=2
    # B2[:, 0:2]: rows 2-3 of B, cols 0-1. b_row_offset=2, b_col_offset=0
    
    # Strassen for A1 @ B1[:, 0:2]
    strassen_terms(0, 0, 0, 0, 0)
    # Strassen for A2 @ B2[:, 0:2]
    strassen_terms(0, 2, 2, 0, 0)
    
    # C[:, 2:4] = A1 @ B1[:, 2:4] + A2 @ B2[:, 2:4]
    # B1[:, 2:4]: rows 0-1 of B, cols 2-3. b_row_offset=0, b_col_offset=2
    # B2[:, 2:4]: rows 2-3 of B, cols 2-3. b_row_offset=2, b_col_offset=2
    
    strassen_terms(0, 0, 0, 2, 2)
    strassen_terms(0, 2, 2, 2, 2)
    
    # C[:, 4] = A1 @ B1[:, 4] + A2 @ B2[:, 4]
    # B1[:, 4]: rows 0-1 of B, col 4. This is 2x1
    # B2[:, 4]: rows 2-3 of B, col 4. This is 2x1
    # (2x2)(2x1): schoolbook rank 4
    # C[0][4] = A[0][0]*B[0][4] + A[0][1]*B[1][4] + A[0][2]*B[2][4] + A[0][3]*B[3][4]
    # C[1][4] = A[1][0]*B[0][4] + A[1][1]*B[1][4] + A[1][2]*B[2][4] + A[1][3]*B[3][4]
    
    # 4 terms for A1 @ B1[:, 4]
    for j in range(2):
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        # A[0][j] * B[j][4]
        u[j] = 1  # A[0][j], row 0, col j
        v[j * p + 4] = 1  # B[j][4]
        w[4 * n + 0] = 1  # C[0][4]
        add_term(u, v, w)
        
        # A[1][j] * B[j][4]
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[m + j] = 1  # A[1][j], row 1, col j
        v[j * p + 4] = 1
        w[4 * n + 1] = 1  # C[1][4]
        add_term(u, v, w)
    
    # 4 terms for A2 @ B2[:, 4]
    for j in range(2, 4):
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[j] = 1  # A[0][j]
        v[j * p + 4] = 1
        w[4 * n + 0] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[m + j] = 1  # A[1][j]
        v[j * p + 4] = 1
        w[4 * n + 1] = 1
        add_term(u, v, w)
    
    rank = len(U_rows)
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
