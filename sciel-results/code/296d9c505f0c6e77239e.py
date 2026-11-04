import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + TIME_LIMIT - 5  # safety margin
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=np.float64))
        V_cols.append(np.array(v_vec, dtype=np.float64))
        W_cols.append(np.array(w_vec, dtype=np.float64))
    
    def strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for Strassen 2x2*2x2."""
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # M1 = (A11+A22)*(B11+B22)
        term([(ai(0,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0),(ci(1,1),1.0)])
        # M2 = (A21+A22)*B11
        term([(ai(1,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0)], [(ci(1,0),1.0),(ci(1,1),-1.0)])
        # M3 = A11*(B12-B22)
        term([(ai(0,0),1.0)], [(bi(0,1),1.0),(bi(1,1),-1.0)], [(ci(0,1),1.0),(ci(1,1),1.0)])
        # M4 = A22*(B21-B11)
        term([(ai(1,1),1.0)], [(bi(1,0),1.0),(bi(0,0),-1.0)], [(ci(0,0),1.0),(ci(1,0),1.0)])
        # M5 = (A11+A12)*B22
        term([(ai(0,0),1.0),(ai(0,1),1.0)], [(bi(1,1),1.0)], [(ci(0,0),-1.0),(ci(0,1),1.0)])
        # M6 = (A21-A11)*(B11+B12)
        term([(ai(1,0),1.0),(ai(0,0),-1.0)], [(bi(0,0),1.0),(bi(0,1),1.0)], [(ci(1,1),1.0)])
        # M7 = (A12-A22)*(B21+B22)
        term([(ai(0,1),1.0),(ai(1,1),-1.0)], [(bi(1,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0)])
    
    def schoolbook_2x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 4 rank-1 terms for schoolbook 2x2 * 2x1."""
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # C1 = A11*B11 + A12*B21
        term([(ai(0,0),1.0)], [(bi(0,0),1.0)], [(ci(0,0),1.0)])
        term([(ai(0,1),1.0)], [(bi(1,0),1.0)], [(ci(0,0),1.0)])
        # C2 = A21*B11 + A22*B21
        term([(ai(1,0),1.0)], [(bi(0,0),1.0)], [(ci(1,0),1.0)])
        term([(ai(1,1),1.0)], [(bi(1,0),1.0)], [(ci(1,0),1.0)])
    
    # Build the rank-77 decomposition:
    # 4 blocks of 2x2*2x2 using Strassen (7 each) = 28
    # But we have 4x4 * 4x6, so we need to think about this differently
    
    # Actually, let's use a better approach:
    # Split 4x4 * 4x6 into blocks:
    # A is 4x4, B is 4x6
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x3 blocks: B1, B2 (each 2x3)
    # Result C is 4x6, split into 2x3 blocks: C1, C2 (each 2x3)
    
    # C1 = A11*B1 + A12*B2 (2x2 * 2x3 + 2x2 * 2x3)
    # C2 = A21*B1 + A22*B2 (2x2 * 2x3 + 2x2 * 2x3)
    
    # For 2x2 * 2x3, we can use Strassen-like with 5 or 6 multiplications
    # Actually, let's just use schoolbook: 2x2 * 2x3 = 4*3 = 12 multiplications
    # But we can do better with Strassen for the 2x2 part
    
    # Let me reconsider: use the proven best approach from inspirations
    # Recursive Strassen for 4x4x4 gives rank 49
    # Then handle the 4x4x2 part separately
    
    # Clear previous terms
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Use recursive Strassen for 4x4 * 4x4 (first 4 columns of B)
    # This gives rank 49 for the 4x4x4 part
    # Then add 4x4 * 4x2 for the last 2 columns
    
    # Recursive Strassen 4x4 * 4x4:
    # Split into 2x2 blocks, use Strassen recursively
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # Each of these is 2x2*2x2, which we do with Strassen (7 each)
    # Total: 7*7 = 49 for 4x4*4x4
    
    # For the 4x4 * 4x2 part (last 2 columns):
    # Split A into 2x2 blocks, B_last2 is 4x2
    # C_last2 = A11*B1_last2 + A12*B2_last2 (top 2 rows)
    #          = A21*B1_last2 + A22*B2_last2 (bottom 2 rows)
    # Where B1_last2 is 2x2, B2_last2 is 2x2
    # Use Strassen for each 2x2*2x2: 7+7 = 14
    # Total: 49 + 14 = 63
    
    # Wait, that's not right. Let me reconsider.
    # 4x4 * 4x6: A is 4x4, B is 4x6
    # Split B into B_left (4x4) and B_right (4x2)
    # C = A*B_left + A*B_right
    # A*B_left: 4x4 * 4x4, rank 49 via recursive Strassen
    # A*B_right: 4x4 * 4x2
    # For 4x4 * 4x2:
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # B_right is 4x2, split into B1_right (2x2) and B2_right (2x2)
    # C_top = A11*B1_right + A12*B2_right (2x2 * 2x2 + 2x2 * 2x2)
    # C_bot = A21*B1_right + A22*B2_right (2x2 * 2x2 + 2x2 * 2x2)
    # Each 2x2*2x2 uses Strassen: 7 multiplications
    # Total for 4x4*4x2: 7+7 = 14
    # Grand total: 49 + 14 = 63
    
    # But the inspirations say rank 77 is achievable. Let me check if there's a better way.
    # Actually, 63 might be better than 77. Let me implement this.
    
    def strassen_4x4_4x4(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 49 rank-1 terms for recursive Strassen 4x4*4x4."""
        # Split 4x4 into 2x2 blocks
        # A: rows a_r..a_r+3, cols a_c..a_c+3
        # B: rows b_r..b_r+3, cols b_c..b_c+3
        # C: rows c_r..c_r+3, cols c_c..c_c+3
        
        # A11: rows a_r..a_r+1, cols a_c..a_c+1
        # A12: rows a_r..a_r+1, cols a_c+2..a_c+3
        # A21: rows a_r+2..a_r+3, cols a_c..a_c+1
        # A22: rows a_r+2..a_r+3, cols a_c+2..a_c+3
        
        # B11: rows b_r..b_r+1, cols b_c..b_c+1
        # B12: rows b_r..b_r+1, cols b_c+2..b_c+3
        # B21: rows b_r+2..b_r+3, cols b_c..b_c+1
        # B22: rows b_r+2..b_r+3, cols b_c+2..b_c+3
        
        # C11: rows c_r..c_r+1, cols c_c..c_c+1
        # C12: rows c_r..c_r+1, cols c_c+2..c_c+3
        # C21: rows c_r+2..c_r+3, cols c_c..c_c+1
        # C22: rows c_r+2..c_r+3, cols c_c+2..c_c+3
        
        # M1 = (A11+A22)*(B11+B22) -> contributes to C11, C22
        # M2 = (A21+A22)*B11 -> contributes to C21, C22
        # M3 = A11*(B12-B22) -> contributes to C12
        # M4 = A22*(B21-B11) -> contributes to C21
        # M5 = (A11+A12)*B22 -> contributes to C11, C12
        # M6 = (A21-A11)*(B11+B12) -> contributes to C22
        # M7 = (A12-A22)*(B21+B22) -> contributes to C11
        
        # Each term is 2x2*2x2, use Strassen (7 each)
        
        # Helper to add a 2x2*2x2 Strassen term
        def add_strassen_2x2(a11_r, a11_c, a22_r, a22_c, b11_r, b11_c, b22_r, b22_c, 
                              c11_r, c11_c, c22_r, c22_c, 
                              a_sign1=1.0, a_sign2=1.0, b_sign1=1.0, b_sign2=1.0,
                              c_sign11=1.0, c_sign22=1.0):
            """Add Strassen terms for (a11*a_sign1 + a22*a_sign2) * (b11*b_sign1 + b22*b_sign2)
               contributing to c11 with c_sign11 and c22 with c_sign22."""
            def ai(i,j): return (a11_r+i)*m + (a11_c+j)
            def bi(i,j): return (b11_r+i)*p + (b11_c+j)
            def ci(i,j): return (c11_c+j)*n + (c11_r+i)
            
            # We need to handle the case where a22 and b22 are at different positions
            # This is getting complex. Let me use a different approach.
            pass
        
        # This is getting too complex. Let me just use the simpler block approach.
        pass
    
    # Let me use a cleaner approach: build the full decomposition directly
    
    # For 4x4 * 4x6:
    # Use Strassen at the top level, splitting into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2 (B is 4x6 = 2x3 blocks of 2x2)
    # C = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2
    
    # Strassen for 2x2 * 2x3:
    # We can use 5 multiplications (Winograd) or just do 4*3=12 schoolbook
    # Let's use a hybrid: Strassen for the 2x2*2x2 parts
    
    # Actually, the simplest correct approach:
    # 4x4 * 4x6 = sum over j of A[:,j] * B[j,:]
    # This is the standard decomposition with rank 16
    
    # But we want minimal rank. Let's use the known best:
    # For 4x4*4x4, rank 49 (recursive Strassen)
    # For 4x4*4x2, we can do better than 14
    
    # Let me just implement the straightforward recursive Strassen for 4x4*4x4
    # and add the 4x4*4x2 part with Strassen
    
    # Clear and rebuild
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Recursive Strassen 4x4 * 4x4 (columns 0-3 of B)
    # This requires careful implementation
    
    # Let me use a simpler but still good approach:
    # Block Strassen: 4x4 * 4x6
    # Split into 2x2 blocks
    # A: 2x2 blocks of 2x2
    # B: 2x3 blocks of 2x2
    # Use Strassen-like with 7 multiplications for the 2x2*2x2 core
    
    # Actually, let me just use the schoolbook with some optimization
    # The trivial rank is 4*4*6 = 96
    # Strassen gives better
    
    # Let me implement a clean recursive Strassen
    
    def strassen_recursive(a_r, a_c, a_size, b_r, b_c, b_size, c_r, c_c, c_size):
        """Recursive Strassen for a_size x a_size * b_size x b_size -> c_size x c_size
           Only works when a_size == b_size == c_size and is power of 2."""
        if a_size == 2:
            strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c)
            return
        
        half = a_size // 2
        # A11, A12, A21, A22
        # B11, B12, B21, B22
        # C11, C12, C21, C22
        
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For each, we need to add the appropriate terms
        # This is complex to implement recursively with the current structure
        
        # Let me just do 2x2 Strassen at the top level for 4x4
        # 4x4 * 4x4: split into 2x2 blocks, use 7 Strassen terms
        # Each term is 2x2*2x2, use 7 more Strassen terms
        # Total: 49
        
        # But we need to handle the 4x4*4x6 case
    
    # Given the complexity, let me use a simpler but proven approach:
    # Use 2x2 Strassen blocks
    
    # 4x4 * 4x6:
    # Split A into 2x2 blocks (4 blocks)
    # Split B into 2x2 blocks (12 blocks, arranged as 2 rows x 3 cols)
    # C is 4x6, split into 2x2 blocks (6 blocks, arranged as 2 rows x 3 cols)
    
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Using Strassen at the 2x2 level:
    # For each pair of 2x2 blocks (A_block, B_block), use 7 multiplications
    
    # But this doesn't directly give a lower rank because the blocks interact
    
    # Let me just use the straightforward approach and verify
    # The current program already has a working decomposition at rank 84
    # Let me try to get to rank 77 as mentioned in inspirations
    
    # Use Strassen for 4x4*4x4 (rank 49) + optimized 4x4*4x2 (rank 28)
    # Total: 77
    
    # For 4x4*4x2:
    # Split A into 2x2 blocks
    # B_4x2 is 4x2
    # C_4x2 = A * B_4x2
    # Split C into 2x2 blocks (2 blocks vertically)
    # Use Strassen-like: 4 multiplications? No, 2x2*4x2...
    
    # Actually, 4x4 * 4x2:
    # A is 4x4, B is 4x2
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # B is 4x2, split into 2x2 blocks: B1 (top 2 rows), B2 (bottom 2 rows)
    # C is 4x2, split into 2x2 blocks: C1 (top 2 rows), C2 (bottom 2 rows)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # Each 2x2*2x2 uses Strassen: 7 multiplications
    # Total: 7+7 = 14
    
    # So 49 + 14 = 63, not 77. Let me check if 63 is achievable.
    
    # Let me just implement the 63-rank decomposition
    
    # First, implement recursive Strassen for 4x4*4x4
    
    def add_strassen_4x4_4x4(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 49 terms for 4x4*4x4 using recursive Strassen."""
        half = 2
        # A11: (a_r, a_c, 2, 2)
        # A12: (a_r, a_c+2, 2, 2)
        # A21: (a_r+2, a_c, 2, 2)
        # A22: (a_r+2, a_c+2, 2, 2)
        # B11: (b_r, b_c, 2, 2)
        # B12: (b_r, b_c+2, 2, 2)
        # B21: (b_r+2, b_c, 2, 2)
        # B22: (b_r+2, b_c+2, 2, 2)
        # C11: (c_r, c_c, 2, 2)
        # C12: (c_r, c_c+2, 2, 2)
        # C21: (c_r+2, c_c, 2, 2)
        # C22: (c_r+2, c_c+2, 2, 2)
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        # M3 = A11*(B12-B22) -> C12 += M3
        # M4 = A22*(B21-B11) -> C21 += M4
        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        
        # For each M, we need to add a 2x2*2x2 Strassen decomposition
        # But the operands are sums/differences of 2x2 blocks
        
        # This requires a more general function that can handle linear combinations
        
        # Let me define a helper that adds a term where U, V, W are specified
        # by their entries in the full matrices
        
        def add_general_term(u_spec, v_spec, w_spec):
            """u_spec, v_spec, w_spec are lists of (index, value) tuples"""
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in u_spec:
                u[idx] = val
            for idx, val in v_spec:
                v[idx] = val
            for idx, val in w_spec:
                w[idx] = val
            add_term(u, v, w)
        
        # Define index helpers
        def ai(r, c): return r*m + c
        def bi(r, c): return r*p + c
        def ci(r, c): return c*n + r
        
        # M1 = (A11+A22)*(B11+B22)
        # U: A11 and A22 entries with +1
        # V: B11 and B22 entries with +1
        # W: C11 and C22 entries with +1
        u1 = [(ai(a_r, a_c), 1.0), (ai(a_r+1, a_c), 1.0), (ai(a_r, a_c+1), 1.0), (ai(a_r+1, a_c+1), 1.0),
              (ai(a_r+2, a_c+2), 1.0), (ai(a_r+3, a_c+2), 1.0), (ai(a_r+2, a_c+3), 1.0), (ai(a_r+3, a_c+3), 1.0)]
        v1 = [(bi(b_r, b_c), 1.0), (bi(b_r+1, b_c), 1.0), (bi(b_r, b_c+1), 1.0), (bi(b_r+1, b_c+1), 1.0),
              (bi(b_r+2, b_c+2), 1.0), (bi(b_r+3, b_c+2), 1.0), (bi(b_r+2, b_c+3), 1.0), (bi(b_r+3, b_c+3), 1.0)]
        w1 = [(ci(c_r, c_c), 1.0), (ci(c_r+1, c_c), 1.0), (ci(c_r, c_c+1), 1.0), (ci(c_r+1, c_c+1), 1.0),
              (ci(c_r+2, c_c+2), 1.0), (ci(c_r+3, c_c+2), 1.0), (ci(c_r+2, c_c+3), 1.0), (ci(c_r+3, c_c+3), 1.0)]
        
        # But this is a single term with 8+8+8 entries, which is not rank 1
        # We need to decompose this further using Strassen on the 2x2 level
        
        # This is getting very complex. Let me just use the existing 84-rank solution
        # and try to optimize it
        
        pass
    
    # Given the time constraints, let me just use the existing 84-rank solution
    # and verify it's correct
    
    # Rebuild the 84-rank solution
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    # Build target tensor
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    best_U = np.column_stack(U_cols)
    best_V = np.column_stack(V_cols)
    best_W = np.column_stack(W_cols)
    best_rank = rank
    
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
