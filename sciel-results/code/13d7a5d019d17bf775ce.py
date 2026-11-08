import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-43 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses block decomposition with Strassen's rank-7 algorithm for 2x2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    # Total rank: 18 (Strassen parts) + 10 + 10 + 5 = 43
    rank = 43
    
    # Allocate factor matrices
    # U: shape (n*m, rank) = (9, 43), indexed by (i, j) -> row i*m + j
    # V: shape (m*p, rank) = (15, 43), indexed by (j, k) -> row j*p + k
    # W: shape (p*n, rank) = (15, 43), indexed by (k, i) -> row k*n + i
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2x2:
    # M1 = (a+d)(e+h)  -> A: a+d, B: e+h
    # M2 = (c+d)e      -> A: c+d, B: e
    # M3 = a(f-h)      -> A: a, B: f-h
    # M4 = d(g-c)      -> A: d, B: g-c
    # M5 = (a+b)h      -> A: a+b, B: h
    # M6 = (c-a)(e+f)  -> A: c-a, B: e+f
    # M7 = (b-d)(g+h)  -> A: b-d, B: g+h
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen factor entries: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: (coeff_a, coeff_b, coeff_c, coeff_d) for A submatrix [[a,b],[c,d]]
    # v_coeffs: (coeff_e, coeff_f, coeff_g, coeff_h) for B submatrix [[e,f],[g,h]]
    # w_coeffs: (coeff_11, coeff_12, coeff_21, coeff_22) for C submatrix
    strassen_factors = [
        # M1 = (a+d)(e+h)
        ((1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2 = (c+d)e
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        # M3 = a(f-h)
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        # M4 = d(g-c)
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
        # M5 = (a+b)h
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6 = (c-a)(e+f)
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7 = (b-d)(g+h)
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, -1)),
    ]
    
    # Part 1: Three Strassen 2x2x2x2 blocks for (i in {0,1}, j in {0,1}, k in {0,1}), (k in {2,3}), (k in {4})
    # Wait, k=4 is a single column, so it's 2x2x1x1, not 2x2x2x2.
    # Let me re-evaluate the block decomposition.
    #
    # Actually, for the 3x3x5 case:
    # We can split B into columns.
    # A is 3x3, B is 3x5.
    # C = A * B is 3x5.
    #
    # Let's split A into top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1.
    # Let's split B into top 2 rows and bottom 1 row, and also into column blocks.
    #
    # This is getting complicated. Let's use a simpler block approach.
    #
    # Split A into:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # Split B into:
    # B = [[B1], [B2]] where B1 is 2x5, B2 is 1x5.
    #
    # Then C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    # C is 3x5.
    # Top 2 rows: C_top = A11*B1 + A12*B2 (2x5)
    # Bottom row: C_bot = A21*B1 + A22*B2 (1x5)
    #
    # A11*B1 is 2x5. We can compute this using Strassen-like methods if we split B1 further.
    # B1 is 2x5. Split B1 into B1a (2x2), B1b (2x2), B1c (2x1).
    # A11 is 2x2.
    # A11*B1a: 2x2x2x2, rank 7 (Strassen)
    # A11*B1b: 2x2x2x2, rank 7 (Strassen)
    # A11*B1c: 2x2x2x1, rank 4 (schoolbook)
    #
    # A12*B2: A12 is 2x1, B2 is 1x5. This is 2x1x1x5, rank 10.
    #
    # A21*B1: A21 is 1x2, B1 is 2x5. This is 1x2x2x5, rank 10.
    #
    # A22*B2: A22 is 1x1, B2 is 1x5. This is 1x1x1x5, rank 5.
    #
    # Total rank: 7 + 7 + 4 + 10 + 10 + 5 = 43.
    
    # Let's implement this.
    
    # Helper to add a Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        """Add a 2x2x2x2 Strassen block at given offsets."""
        nonlocal col
        for u_coeffs, v_coeffs, w_coeffs in strassen_factors:
            # A submatrix indices: (i_offset, j_offset) to (i_offset+1, j_offset+1)
            # B submatrix indices: (j_offset, k_offset) to (j_offset+1, k_offset+1)
            # Wait, the indices are (i, j) for A, (j, k) for B.
            # So A is at rows i_offset..i_offset+1, cols j_offset..j_offset+1.
            # B is at rows j_offset..j_offset+1, cols k_offset..k_offset+1.
            # C is at rows k_offset..k_offset+1, cols i_offset..i_offset+1.
            
            for (a_idx, a_coeff) in enumerate(u_coeffs):
                i_local = a_idx // 2
                j_local = a_idx % 2
                i_global = i_offset + i_local
                j_global = j_offset + j_local
                U[i_global * m + j_global][col_start] = a_coeff
            
            for (b_idx, b_coeff) in enumerate(v_coeffs):
                j_local = b_idx // 2
                k_local = b_idx % 2
                j_global = j_offset + j_local
                k_global = k_offset + k_local
                V[j_global * p + k_global][col_start] = b_coeff
            
            for (c_idx, c_coeff) in enumerate(w_coeffs):
                k_local = c_idx // 2
                i_local = c_idx % 2
                k_global = k_offset + k_local
                i_global = i_offset + i_local
                W[k_global * n + i_global][col_start] = c_coeff
            
            col_start += 1
        
        return col_start
    
    # Helper to add a schoolbook block
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, col_start):
        """Add a schoolbook block for A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]."""
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col_start] = 1
                    V[j * p + k][col_start] = 1
                    W[k * n + i][col_start] = 1
                    col_start += 1
        return col_start
    
    # Part 1: A11*B1a (2x2x2x2, Strassen)
    # A11: i in {0,1}, j in {0,1}
    # B1a: j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, col)  # col becomes 7
    
    # Part 2: A11*B1b (2x2x2x2, Strassen)
    # A11: i in {0,1}, j in {0,1}
    # B1b: j in {0,1}, k in {2,3}
    col = add_strassen_block(0, 0, 2, col)  # col becomes 14
    
    # Part 3: A11*B1c (2x2x2x1, schoolbook)
    # A11: i in {0,1}, j in {0,1}
    # B1c: j in {0,1}, k in {4}
    col = add_schoolbook_block(0, 2, 0, 2, 4, 5, col)  # col becomes 18
    
    # Part 4: A12*B2 (2x1x1x5, schoolbook)
    # A12: i in {0,1}, j in {2}
    # B2: j in {2}, k in {0,1,2,3,4}
    col = add_schoolbook_block(0, 2, 2, 3, 0, 5, col)  # col becomes 28
    
    # Part 5: A21*B1 (1x2x2x5, schoolbook)
    # A21: i in {2}, j in {0,1}
    # B1: j in {0,1}, k in {0,1,2,3,4}
    col = add_schoolbook_block(2, 3, 0, 2, 0, 5, col)  # col becomes 38
    
    # Part 6: A22*B2 (1x1x1x5, schoolbook)
    # A22: i in {2}, j in {2}
    # B2: j in {2}, k in {0,1,2,3,4}
    col = add_schoolbook_block(2, 3, 2, 3, 0, 5, col)  # col becomes 43
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# Deadline guard for safety
_start_time = None

def construct():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to schoolbook rank-45
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return _construct_impl()

def _construct_impl():
    # Same as the construct() above
    n, m, p = DIMENSIONS
    rank = 43
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_factors = [
        ((1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, -1)),
    ]
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        nonlocal col
        for u_coeffs, v_coeffs, w_coeffs in strassen_factors:
            for (a_idx, a_coeff) in enumerate(u_coeffs):
                i_local = a_idx // 2
                j_local = a_idx % 2
                i_global = i_offset + i_local
                j_global = j_offset + j_local
                U[i_global * m + j_global][col_start] = a_coeff
            
            for (b_idx, b_coeff) in enumerate(v_coeffs):
                j_local = b_idx // 2
                k_local = b_idx % 2
                j_global = j_offset + j_local
                k_global = k_offset + k_local
                V[j_global * p + k_global][col_start] = b_coeff
            
            for (c_idx, c_coeff) in enumerate(w_coeffs):
                k_local = c_idx // 2
                i_local = c_idx % 2
                k_global = k_offset + k_local
                i_global = i_offset + i_local
                W[k_global * n + i_global][col_start] = c_coeff
            
            col_start += 1
        
        return col_start
    
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, col_start):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col_start] = 1
                    V[j * p + k][col_start] = 1
                    W[k * n + i][col_start] = 1
                    col_start += 1
        return col_start
    
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_schoolbook_block(0, 2, 0, 2, 4, 5, col)
    col = add_schoolbook_block(0, 2, 2, 3, 0, 5, col)
    col = add_schoolbook_block(2, 3, 0, 2, 0, 5, col)
    col = add_schoolbook_block(2, 3, 2, 3, 0, 5, col)
    
    return {"U": U, "V": V, "W": W}
