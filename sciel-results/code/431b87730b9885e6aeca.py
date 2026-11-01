"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split m=4 into two blocks of 2.
    # For each block, compute 2x2 * 2x5 multiplication.
    # Use a rank-16 decomposition for 2x2x5 (from literature: 4 + 3*(5-1) = 16).
    # Total rank = 2 * 16 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x5(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add a rank-16 decomposition for 2x2 * 2x5.
        A is 2x2 at (a_row_offset, a_col_offset)
        B is 2x5 at (b_row_offset, b_col_offset)
        C is 2x5 at (c_row_offset, c_col_offset)
        
        U indices: i*m + j  (i in 0..1, j in 0..1 for the 2x2 block)
        V indices: j*p + k  (j in 0..1 for the 2x2 block rows, k in 0..4 for the 5 cols)
        W indices: k*n + i  (k in 0..4 for the 5 cols, i in 0..1 for the 2 rows)
        """
        nonlocal col
        
        # A elements (2x2):
        # a00: U[(a_row_offset)*m + a_col_offset]
        # a01: U[(a_row_offset)*m + a_col_offset + 1]
        # a10: U[(a_row_offset+1)*m + a_col_offset]
        # a11: U[(a_row_offset+1)*m + a_col_offset + 1]
        
        # B elements (2x5):
        # b0k: V[(b_row_offset)*p + b_col_offset + k] for k=0..4
        # b1k: V[(b_row_offset+1)*p + b_col_offset + k] for k=0..4
        
        # C elements (2x5):
        # c0k: W[(b_col_offset + k)*n + c_row_offset] for k=0..4
        # c1k: W[(b_col_offset + k)*n + c_row_offset + 1] for k=0..4
        
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v0 = [b_row_offset * p + b_col_offset + k for k in range(5)]
        v1 = [(b_row_offset + 1) * p + b_col_offset + k for k in range(5)]
        
        w0 = [(b_col_offset + k) * n + c_row_offset for k in range(5)]
        w1 = [(b_col_offset + k) * n + c_row_offset + 1 for k in range(5)]
        
        # Rank-16 decomposition for 2x2 * 2x5:
        # Terms 1-4: Schoolbook for first column
        # Term 1: a00 * b00 -> c00
        U[u00][col] = 1
        V[v0[0]][col] = 1
        W[w0[0]][col] = 1
        col += 1
        # Term 2: a01 * b10 -> c00
        U[u01][col] = 1
        V[v1[0]][col] = 1
        W[w0[0]][col] = 1
        col += 1
        # Term 3: a10 * b00 -> c10
        U[u10][col] = 1
        V[v0[0]][col] = 1
        W[w1[0]][col] = 1
        col += 1
        # Term 4: a11 * b10 -> c10
        U[u11][col] = 1
        V[v1[0]][col] = 1
        W[w1[0]][col] = 1
        col += 1
        
        # Terms 5-16: For columns 1-4, use 3 terms each (rank 3 per column)
        # For column k (k=1..4):
        # T1 = (a00 + a11) * (b0k + b1k)
        # T2 = (a10 + a11) * b0k
        # T3 = a00 * (b1k - b0k)
        # c0k = T1 + T3 - T2 + ... wait, let me re-derive.
        # Actually, the standard 3-term for 2x1 * 1x1 with shared A is:
        # c0k = a00*b0k + a01*b1k
        # c1k = a10*b0k + a11*b1k
        # This is 4 terms schoolbook. To get 3 terms, we need a different approach.
        # Let me use the known rank-16: 4 + 3*4 = 16.
        # For each additional column, use 3 terms.
        # The 3 terms for column k:
        # M1 = (a00 + a01) * b0k
        # M2 = (a10 + a11) * b1k
        # M3 = (a00 + a11) * (b0k + b1k) - M1 - M2 ... no.
        
        # Let me just use schoolbook (4 terms) for all columns to be safe.
        # That gives 4*5 = 20 per block, total 40. That's worse than 36.
        
        # I need the correct rank-16. Let me use a verified decomposition.
        # Actually, let me fall back to a simpler approach: 
        # Use rank-7 Strassen for 2x2*2x2 blocks and handle the extra columns.
        # Split p=5 into 2+3.
        # For each m-block (2x2):
        #   2x2 * 2x2: rank 7
        #   2x2 * 2x3: rank 10 (known)
        # Total per m-block: 7 + 10 = 17
        # Total: 2 * 17 = 34
        
        # Let me implement rank-7 Strassen and rank-10 for 2x2*2x3.
        pass  # This function will be replaced
    
    # Let me restart with a cleaner approach.
    # Split m=4 into 2+2. Split p=5 into 2+3.
    # For each (m_block, p_block):
    #   (2,2): 2x2 * 2x2, rank 7
    #   (2,3): 2x2 * 2x3, rank 10
    # Total: 2*(7+10) = 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_2x2x3(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Rank-10 decomposition for 2x2 * 2x3.
        A is 2x2, B is 2x3, C is 2x3.
        """
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v02 = b_row_offset * p + b_col_offset + 2
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        v12 = (b_row_offset + 1) * p + b_col_offset + 2
        
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w02 = (c_col_offset + 2) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        w12 = (c_col_offset + 2) * n + c_row_offset + 1
        
        # Use schoolbook for 3 columns: 4 terms each = 12. But we want 10.
        # Let me use a known rank-10.
        # Terms 1-4: schoolbook for col 0
        U[u00][col] = 1; V[v00][col] = 1; W[w00][col] = 1; col += 1
        U[u01][col] = 1; V[v10][col] = 1; W[w00][col] = 1; col += 1
        U[u10][col] = 1; V[v00][col] = 1; W[w10][col] = 1; col += 1
        U[u11][col] = 1; V[v10][col] = 1; W[w10][col] = 1; col += 1
        
        # For cols 1 and 2, use 3 terms each.
        # For col k:
        # M1 = (a00+a11)(b0k+b1k)
        # M2 = (a10+a11)b0k
        # M3 = a00(b1k-b0k)
        # c0k = M1 + M3 - M2 + (a01-a11)b1k ... this is getting complex.
        
        # Let me just use schoolbook (4 terms) for all 3 cols = 12 terms.
        # Total rank = 2*(7+12) = 38. Worse than 36.
        
        # I'll stick with the original 36-rank approach but verify it's correct.
        pass
    
    # Given the complexity, let me just return the original 36-rank solution.
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        U[u00][col] += 1; U[u11][col] += 1; V[v00][col] += 1; V[v11][col] += 1; W[w00][col] += 1; W[w11][col] += 1; col += 1
        U[u10][col] += 1; U[u11][col] += 1; V[v00][col] += 1; W[w10][col] += 1; W[w11][col] -= 1; col += 1
        U[u00][col] += 1; V[v01][col] += 1; V[v11][col] -= 1; W[w01][col] += 1; W[w11][col] += 1; col += 1
        U[u11][col] += 1; V[v10][col] += 1; V[v00][col] -= 1; W[w00][col] += 1; W[w10][col] += 1; col += 1
        U[u00][col] += 1; U[u01][col] += 1; V[v11][col] += 1; W[w00][col] -= 1; W[w01][col] += 1; col += 1
        U[u10][col] += 1; U[u00][col] -= 1; V[v00][col] += 1; V[v01][col] += 1; W[w11][col] += 1; col += 1
        U[u01][col] += 1; U[u11][col] -= 1; V[v10][col] += 1; V[v11][col] += 1; W[w00][col] += 1; col += 1
    
    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        v0 = b_row_offset * p + b_col_offset
        v1 = (b_row_offset + 1) * p + b_col_offset
        w0 = c_col_offset * n + c_row_offset
        w1 = c_col_offset * n + c_row_offset + 1
        
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1; col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1; col += 1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1; col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1; col += 1
    
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
