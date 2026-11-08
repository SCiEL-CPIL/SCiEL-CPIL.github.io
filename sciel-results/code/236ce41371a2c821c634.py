"""Matrix multiplication 2,4,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    # We split m=4 into two blocks of 2.
    # For each 2x2x7 block, we split p=7 into 2+2+3.
    # 2x2x2: rank 7 (Strassen)
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: rank 6 (trivial: 2*2*3)
    # Total per block: 7+7+6 = 20... that's too high.
    # Actually 2x2x3 can be done in rank 6 trivially, but let me reconsider.
    # 
    # Better approach: 2x2x7. The 2x2x7 multiplication:
    # C (2x7) = A (2x2) * B (2x7)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # We can use rank-9 for 2x2x7:
    # Split B's 7 columns into groups of 2,2,3.
    # For the 2+2 part: use Strassen-like for 2x2x2 (rank 7 each)
    # For the 3 part: trivial 2x2x3 (rank 6)
    # That gives 7+7+6=20 per block, total 40. Still not minimal.
    #
    # Let me think more carefully. The known bound for 2x2x7:
    # Actually, for 2x2xp, the rank is at most 2*2*p = 4p trivially.
    # But we can do better. For 2x2x7, rank <= 11 or so.
    #
    # Let me just use a straightforward but efficient decomposition:
    # Split m=4 into 2+2. For each 2x2x7 block:
    #   Use the fact that 2x2x7 has rank at most 12 (or similar).
    #
    # Actually, let me just build the trivial-but-optimized version:
    # For 2x2x7: C[i,k] = sum_j A[i,j]*B[j,k] for j in {0,1}
    # This is just 2*2*7 = 28 terms trivially.
    # But we can reduce: note that for fixed i, C[i,k] depends on
    # A[i,0] and A[i,1] times the rows of B.
    #
    # Let me use a known approach: 2x2x7 can be done in rank 11.
    # The idea: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Define for the 2x2 part:
    # M0 = A[i,0]*B[0,k] -> this is rank 7 (one per k)
    # M1 = A[i,1]*B[1,k] -> rank 7
    # Total 14. Can we do better?
    #
    # Actually the simplest correct approach: total rank = 2*2*7 + 2*2*7 = 56 trivially.
    # But we can share structure. Let me just do the direct construction
    # and accept a moderate rank improvement.
    #
    # Key optimization: For 2x2xp, we can use rank 2p-1 via the identity:
    # A[i,0]*B[0,k] + A[i,1]*B[1,k] = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) 
    #   - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    # Hmm, that doesn't directly help for different k.
    #
    # Let me just use the straightforward block decomposition with Strassen for 2x2x2 parts:
    # Split 2x4x7 as two 2x2x7 blocks.
    # For 2x2x7: split 7 into 2+2+3.
    #   2x2x2: Strassen rank 7
    #   2x2x2: Strassen rank 7  
    #   2x2x3: trivial rank 6
    #   Total: 20 per block, 40 total.
    #
    # That's better than 56. Let me implement this.
    
    rank = 40
    U = [[0]*rank for _ in range(n*m)]  # 8 x 40
    V = [[0]*rank for _ in range(m*p)]  # 28 x 40
    W = [[0]*rank for _ in range(p*n)]  # 14 x 40
    
    col = 0
    
    def strassen_2x2x2(u_offset, v_offset, w_offset, start_col):
        """Add Strassen decomposition for 2x2x2 block.
        u_offset: row offset in U (0-1 for A indices)
        v_offset: row offset in V (0-1 for B row indices, each with 2 cols)
        w_offset: row offset in W (0-1 for C row indices, each with 2 cols)
        Returns: next column index
        """
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U indices: A[0,0]=u_offset+0, A[0,1]=u_offset+1, A[1,0]=u_offset+2, A[1,1]=u_offset+3
        # V indices: B[0,0]=v_offset+0, B[0,1]=v_offset+1, B[1,0]=v_offset+2, B[1,1]=v_offset+3
        # W indices: C[0,0]=w_offset+0, C[0,1]=w_offset+1, C[1,0]=w_offset+2, C[1,1]=w_offset+3
        
        # Wait, I need to be careful with indexing conventions.
        # U[i*m+j] where i is row of A, j is col of A. For 2x2 block starting at col c0:
        #   A[0,c0] -> U[row0], A[0,c0+1] -> U[row0+1]
        #   A[1,c0] -> U[row1], A[1,c0+1] -> U[row1+1]
        # For n=2, m=4: U index = i*4+j
        #
        # Let me re-parameterize. I'll pass explicit U row indices, V row indices, W row indices.
        
        nonlocal col
        # This function is getting complex. Let me inline the Strassen terms.
        pass
    
    # Let me take a simpler, more explicit approach.
    # Build the whole thing step by step.
    
    # Clear and rebuild
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen helper for 2x2x2
    # A rows: a0, a1 (these are U row indices)
    # A cols within block: col_a0, col_a1 (U column positions, but we use rows)
    # Actually U[i*m+j] is the entry. For a 2x2 subblock of A at rows (r0,r1) and cols (c0,c1):
    #   U[r0*m+c0], U[r0*m+c1], U[r1*m+c0], U[r1*m+c1]
    # B rows: b0, b1, cols: cc0, cc1 (within the 2x2x2, B is 2x2)
    #   V[b0*p+cc0], V[b0*p+cc1], V[b1*p+cc0], V[b1*p+cc1]
    # C rows: r0, r1, cols: cc0, cc1
    #   W[cc0*n+r0], W[cc0*n+r1], W[cc1*n+r0], W[cc1*n+r1]
    
    def add_strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A[i,j] maps to U[A_rows[i]*m + A_cols[j]]
        # B[i,j] maps to V[B_rows[i]*p + B_cols[j]]
        # C[i,j] maps to W[C_cols[j]*n + C_rows[i]]
        
        U_A = [[U[A_rows[i]*m + A_cols[j]] for j in range(2)] for i in range(2)]
        V_B = [[V[B_rows[i]*p + B_cols[j]] for j in range(2)] for i in range(2)]
        W_C = [[W[C_cols[j]*n + C_rows[i]] for j in range(2)] for i in range(2)]
        
        # M1 = (A00+A11)(B00+B11)
        def add_term(u_vec, v_vec, w_vec):
            nonlocal col
            for ui in range(2):
                for uj in range(2):
                    if u_vec[ui][uj] != 0:
                        U[A_rows[ui]*m + A_cols[uj]][col] += u_vec[ui][uj]
            for vi in range(2):
                for vj in range(2):
                    if v_vec[vi][vj] != 0:
                        V[B_rows[vi]*p + B_cols[vj]][col] += v_vec[vi][vj]
            for wi in range(2):
                for wj in range(2):
                    if w_vec[wi][wj] != 0:
                        W[C_cols[wj]*n + C_rows[wi]][col] += w_vec[wi][wj]
            col += 1
        
        # M1
        add_term([[1,0],[0,1]], [[1,1],[1,1]], [[1,0],[0,1]])
        # M2
        add_term([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]])
        # M3
        add_term([[1,0],[0,0]], [[0,0],[1,-1]], [[0,0],[0,1]])
        # M4
        add_term([[0,0],[0,1]], [[1,-1],[0,0]], [[1,0],[0,1]])
        # M5
        add_term([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[1,1]])
        # M6
        add_term([[0,0],[1,-1]], [[1,1],[0,0]], [[0,0],[0,0]])  # Wait, M6 = (A10-A00)(B00+B10)
        # Let me redo this more carefully.
        
        # Actually, let me just hardcode the 7 Strassen terms directly.
        # The standard Strassen for C = A*B where all are 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        # c00 = m1+m4-m5+m7
        # c01 = m3+m5
        # c10 = m2+m4
        # c11 = m1-m2+m3+m6
        
        # Each mi is a rank-1 term: u_i * v_i * w_i
        # where u_i is 2x2 (coefficients for A), v_i is 2x2 (coefficients for B), w_i is 2x2 (coefficients for C)
        
        # m1: u=[[1,0],[0,1]], v=[[1,1],[1,1]], w=[[1,0],[0,1]]
        # m2: u=[[0,0],[1,1]], v=[[1,0],[0,0]], w=[[0,0],[1,0]]
        # m3: u=[[1,0],[0,0]], v=[[0,0],[1,-1]], w=[[0,0],[0,1]]
        # m4: u=[[0,0],[0,1]], v=[[1,-1],[0,0]], w=[[1,0],[0,1]]
        # m5: u=[[1,1],[0,0]], v=[[0,0],[0,1]], w=[[0,0],[1,1]]
        # m6: u=[[0,0],[1,-1]], v=[[1,1],[0,0]], w=[[0,0],[0,0]] -> NO, w for m6 goes to c11
        # m6: u=[[0,0],[1,-1]], v=[[1,1],[0,0]], w=[[0,0],[0,1]]  (contributes to c11)
        # m7: u=[[0,1],[0,-1]], v=[[0,1],[1,1]], w=[[1,0],[0,0]]  (contributes to c00)
        
        terms = [
            ([[1,0],[0,1]], [[1,1],[1,1]], [[1,0],[0,1]]),  # m1
            ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),   # m2
            ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,0],[0,1]]),  # m3
            ([[0,0],[0,1]], [[1,-1],[0,0]], [[1,0],[0,1]]),  # m4
            ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[1,1]]),   # m5
            ([[0,0],[1,-1]], [[1,1],[0,0]], [[0,0],[0,1]]),  # m6
            ([[0,1],[0,-1]], [[0,1],[1,1]], [[1,0],[0,0]]),  # m7
        ]
        
        for u_mat, v_mat, w_mat in terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        U[A_rows[ii]*m + A_cols[jj]][col] += u_mat[ii][jj]
                    if v_mat[ii][jj] != 0:
                        V[B_rows[ii]*p + B_cols[jj]][col] += v_mat[ii][jj]
                    if w_mat[ii][jj] != 0:
                        W[C_cols[jj]*n + C_rows[ii]][col] += w_mat[ii][jj]
            col += 1
    
    # Trivial 2x2x3 helper
    def add_trivial_2x2x3(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[A_rows[i]*m + A_cols[j]][col] = 1
                    V[B_rows[j]*p + B_cols[k]][col] = 1
                    W[C_cols[k]*n + C_rows[i]][col] = 1
                    col += 1
    
    # Now build the decomposition:
    # Block 1: A cols 0,1; B rows 0,1; C all
    #   Split C cols into 0,1 / 2,3 / 4,5,6
    #   2x2x2: Strassen
    #   2x2x2: Strassen
    #   2x2x3: Trivial
    add_strassen_2x2x2(
        A_rows=[0,1], A_cols=[0,1],
        B_rows=[0,1], B_cols=[0,1],
        C_rows=[0,1], C_cols=[0,1]
    )
    add_strassen_2x2x2(
        A_rows=[0,1], A_cols=[0,1],
        B_rows=[0,1], B_cols=[2,3],
        C_rows=[0,1], C_cols=[2,3]
    )
    add_trivial_2x2x3(
        A_rows=[0,1], A_cols=[0,1],
        B_rows=[0,1], B_cols=[4,5,6],
        C_rows=[0,1], C_cols=[4,5,6]
    )
    
    # Block 2: A cols 2,3; B rows 2,3; C all
    add_strassen_2x2x2(
        A_rows=[0,1], A_cols=[2,3],
        B_rows=[2,3], B_cols=[0,1],
        C_rows=[0,1], C_cols=[0,1]
    )
    add_strassen_2x2x2(
        A_rows=[0,1], A_cols=[2,3],
        B_rows=[2,3], B_cols=[2,3],
        C_rows=[0,1], C_cols=[2,3]
    )
    add_trivial_2x2x3(
        A_rows=[0,1], A_cols=[2,3],
        B_rows=[2,3], B_cols=[4,5,6],
        C_rows=[0,1], C_cols=[4,5,6]
    )
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
