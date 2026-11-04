"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices:
    # A: (i, j) where i in 0..1, j in 0..1
    # B: (j, k) where j in 0..1, k in 0..1
    # C: (k, i) where k in 0..1, i in 0..1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_strassen(offset_u, offset_v, offset_w, start_col):
        # offset_u: row offset in U
        # offset_v: col offset in V (actually row offset in V, which corresponds to j*p+k)
        # offset_w: row offset in W
        # We need to map the 2x2 blocks to the global indices.
        
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will add 7 terms.
        # Each term has coefficients in U, V, W.
        
        # Term 1: M1
        # U: A11(1,1) + A22(2,2) -> (offset_u+0, offset_v? No, U is i*m+j)
        # U indices: (0,0) and (1,1) relative to block.
        # V indices: (0,0) and (1,1) relative to block.
        # W indices: (0,0) and (1,1) relative to block.
        
        # Let's define the local indices for the 2x2 blocks.
        # A block rows: 0, 1. Cols: 0, 1.
        # B block rows: 0, 1. Cols: 0, 1.
        # C block rows: 0, 1. Cols: 0, 1.
        
        # Global U index: offset_u + i*m + j
        # Global V index: offset_v + j*p + k
        # Global W index: offset_w + k*n + i
        
        # M1: (A11+A22)(B11+B22)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,1):1
        U[offset_u + 0*m + 0][start_col] = 1
        U[offset_u + 1*m + 1][start_col] = 1
        V[offset_v + 0*p + 0][start_col] = 1
        V[offset_v + 1*p + 1][start_col] = 1
        W[offset_w + 0*n + 0][start_col] = 1
        W[offset_w + 1*n + 1][start_col] = 1
        col = start_col + 1
        
        # M2: (A21+A22)B11
        # U: (1,0):1, (1,1):1
        # V: (0,0):1
        # W: (0,1):1
        U[offset_u + 1*m + 0][col] = 1
        U[offset_u + 1*m + 1][col] = 1
        V[offset_v + 0*p + 0][col] = 1
        W[offset_w + 0*n + 1][col] = 1
        col += 1
        
        # M3: A11(B21-B22)
        # U: (0,0):1
        # V: (1,0):1, (1,1):-1
        # W: (0,0):1
        U[offset_u + 0*m + 0][col] = 1
        V[offset_v + 1*p + 0][col] = 1
        V[offset_v + 1*p + 1][col] = -1
        W[offset_w + 0*n + 0][col] = 1
        col += 1
        
        # M4: A22(B12-B11)
        # U: (1,1):1
        # V: (0,1):1, (0,0):-1
        # W: (1,0):1
        U[offset_u + 1*m + 1][col] = 1
        V[offset_v + 0*p + 1][col] = 1
        V[offset_v + 0*p + 0][col] = -1
        W[offset_w + 1*n + 0][col] = 1
        col += 1
        
        # M5: (A11+A12)B22
        # U: (0,0):1, (0,1):1
        # V: (1,1):1
        # W: (1,1):1
        U[offset_u + 0*m + 0][col] = 1
        U[offset_u + 0*m + 1][col] = 1
        V[offset_v + 1*p + 1][col] = 1
        W[offset_w + 1*n + 1][col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        # U: (1,0):1, (0,0):-1
        # V: (0,0):1, (0,1):1
        # W: (1,1):1
        U[offset_u + 1*m + 0][col] = 1
        U[offset_u + 0*m + 0][col] = -1
        V[offset_v + 0*p + 0][col] = 1
        V[offset_v + 0*p + 1][col] = 1
        W[offset_w + 1*n + 1][col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        # U: (0,1):1, (1,1):-1
        # V: (1,0):1, (1,1):1
        # W: (0,0):1
        U[offset_u + 0*m + 1][col] = 1
        U[offset_u + 1*m + 1][col] = -1
        V[offset_v + 1*p + 0][col] = 1
        V[offset_v + 1*p + 1][col] = 1
        W[offset_w + 0*n + 0][col] = 1
        col += 1
        
        return col

    # Helper to add schoolbook term
    def add_schoolbook(i, j, k, start_col):
        U[i*m + j][start_col] = 1
        V[j*p + k][start_col] = 1
        W[k*n + i][start_col] = 1
        return start_col + 1

    # 1. A1 B1 (2x4 * 4x4)
    # A1 rows 0-1. B1 cols 0-3.
    # Split B1 into B1_left (cols 0-1) and B1_right (cols 2-3).
    # A1 B1_left: 2x4 * 4x2.
    # This is a 2x2 block of Strassen? No, 2x4 * 4x2.
    # A1 = [A11 A12] (2x2, 2x2).
    # B1_left = [B11; B21] (2x2, 2x2).
    # This is ONE Strassen multiplication?
    # No, Strassen is for 2x2 * 2x2.
    # Here we have 2x4 * 4x2.
    # We can split A1 into A11 (2x2) and A12 (2x2).
    # Split B1_left into B11 (2x2) and B21 (2x2).
    # Product = A11 B11 + A12 B21.
    # This is sum of two 2x2x2 tensors.
    # We need 2 Strassen decompositions.
    
    # Strassen 1: A11 * B11
    # A11 is rows 0-1, cols 0-1.
    # B11 is rows 0-1, cols 0-1.
    # C11 is rows 0-1, cols 0-1.
    col = add_strassen(0, 0, 0, col)
    
    # Strassen 2: A12 * B21
    # A12 is rows 0-1, cols 2-3.
    # B21 is rows 2-3, cols 0-1.
    # C12 is rows 0-1, cols 0-1? No.
    # A1 B1_left = [A11 A12] * [B11; B21] = A11 B11 + A12 B21.
    # The result is 2x2.
    # So both terms contribute to C[0:2, 0:2].
    # Wait, the W matrix accumulates.
    # In my add_strassen, W is added to.
    # So I just need to call add_strassen for A12*B21 with correct offsets.
    # A12 offset_u = 0, offset_col = 2.
    # B21 offset_v = 2, offset_col = 0.
    # C offset_w = 0.
    
    # But add_strassen assumes 0-based local indices.
    # I need to modify add_strassen to take offsets for A, B, C.
    
    # Let's rewrite add_strassen to accept offsets.
    
    # Re-define add_strassen with offsets
    def add_strassen_off(u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off, start_col):
        # u_row_off, u_col_off: offset in U (i, j)
        # v_row_off, v_col_off: offset in V (j, k)
        # w_row_off, w_col_off: offset in W (k, i)
        
        # M1
        U[u_row_off + 0*m + u_col_off + 0][start_col] = 1
        U[u_row_off + 1*m + u_col_off + 1][start_col] = 1
        V[v_row_off + 0*p + v_col_off + 0][start_col] = 1
        V[v_row_off + 1*p + v_col_off + 1][start_col] = 1
        W[w_row_off + 0*n + w_col_off + 0][start_col] = 1
        W[w_row_off + 1*n + w_col_off + 1][start_col] = 1
        col = start_col + 1
        
        # M2
        U[u_row_off + 1*m + u_col_off + 0][col] = 1
        U[u_row_off + 1*m + u_col_off + 1][col] = 1
        V[v_row_off + 0*p + v_col_off + 0][col] = 1
        W[w_row_off + 0*n + w_col_off + 1][col] = 1
        col += 1
        
        # M3
        U[u_row_off + 0*m + u_col_off + 0][col] = 1
        V[v_row_off + 1*p + v_col_off + 0][col] = 1
        V[v_row_off + 1*p + v_col_off + 1][col] = -1
        W[w_row_off + 0*n + w_col_off + 0][col] = 1
        col += 1
        
        # M4
        U[u_row_off + 1*m + u_col_off + 1][col] = 1
        V[v_row_off + 0*p + v_col_off + 1][col] = 1
        V[v_row_off + 0*p + v_col_off + 0][col] = -1
        W[w_row_off + 1*n + w_col_off + 0][col] = 1
        col += 1
        
        # M5
        U[u_row_off + 0*m + u_col_off + 0][col] = 1
        U[u_row_off + 0*m + u_col_off + 1][col] = 1
        V[v_row_off + 1*p + v_col_off + 1][col] = 1
        W[w_row_off + 1*n + w_col_off + 1][col] = 1
        col += 1
        
        # M6
        U[u_row_off + 1*m + u_col_off + 0][col] = 1
        U[u_row_off + 0*m + u_col_off + 0][col] = -1
        V[v_row_off + 0*p + v_col_off + 0][col] = 1
        V[v_row_off + 0*p + v_col_off + 1][col] = 1
        W[w_row_off + 1*n + w_col_off + 1][col] = 1
        col += 1
        
        # M7
        U[u_row_off + 0*m + u_col_off + 1][col] = 1
        U[u_row_off + 1*m + u_col_off + 1][col] = -1
        V[v_row_off + 1*p + v_col_off + 0][col] = 1
        V[v_row_off + 1*p + v_col_off + 1][col] = 1
        W[w_row_off + 0*n + w_col_off + 0][col] = 1
        col += 1
        
        return col

    # 1. A1 B1
    # A11 (0,0) * B11 (0,0) -> C11 (0,0)
    col = add_strassen_off(0, 0, 0, 0, 0, 0, col)
    
    # A12 (0,2) * B21 (2,0) -> C11 (0,0)
    col = add_strassen_off(0, 2, 2, 0, 0, 0, col)
    
    # A11 (0,0) * B12 (0,2) -> C12 (0,2)
    col = add_strassen_off(0, 0, 0, 2, 0, 2, col)
    
    # A12 (0,2) * B22 (2,2) -> C12 (0,2)
    col = add_strassen_off(0, 2, 2, 2, 0, 2, col)
    
    # 2. A1 B2a (2x4 * 4x2)
    # B2a is cols 4-5.
    # A11 (0,0) * B21 (0,4) -> C21 (0,4)
    col = add_strassen_off(0, 0, 0, 4, 0, 4, col)
    
    # A12 (0,2) * B22 (2,4) -> C21 (0,4)
    col = add_strassen_off(0, 2, 2, 4, 0, 4, col)
    
    # 3. A1 B2b (2x4 * 4x1)
    # Schoolbook
    for i in range(2):
        for j in range(4):
            for k in range(1):
                col = add_schoolbook(i, j, 4+k, col)
                
    # 4. A2 B1 (1x4 * 4x4)
    # Schoolbook
    for i in range(1):
        i += 2
        for j in range(4):
            for k in range(4):
                col = add_schoolbook(i, j, k, col)
                
    # 5. A2 B2 (1x4 * 4x3)
    # Schoolbook
    for i in range(1):
        i += 2
        for j in range(4):
            for k in range(3):
                col = add_schoolbook(i, j, 4+k, col)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
