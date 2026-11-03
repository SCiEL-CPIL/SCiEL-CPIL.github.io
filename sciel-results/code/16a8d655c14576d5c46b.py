"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-50 tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: 
    - C = A * B where A is 2x4, B is 4x7, C is 2x7
    - Partition A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x7)
    - C = A1*B1 + A2*B2
    - For each 2x2 * 2x7: split 7 cols into 3+4
      - 2x2 * 2x3: Strassen (rank 7) for first 2 cols + schoolbook (rank 4) for last col = rank 11
      - 2x2 * 2x4: two Strassen 2x2*2x2 blocks = rank 14
      - Total per 2x2*2x7: rank 25
    - Total: 2 * 25 = rank 50
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)  
    # W: (p*n, rank) = (14, rank)
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 28 rows
    W = [[0] * rank for _ in range(p * n)]   # 14 rows
    
    col = 0
    
    def add_strassen_222(A_row_offset, B_row_offset, C_row_offset, coeff=1):
        """
        Add a Strassen 2x2x2 block.
        A is 2x2, B is 2x2, C is 2x2.
        A rows: A_row_offset + i (i=0,1), each with 2 cols (j=0,1)
        B rows: B_row_offset + j (j=0,1), each with 2 cols (k=0,1)
        C rows: C_row_offset + k*n + i (k=0,1; i=0,1)
        
        In our indexing:
        U row for A[i,j] = i*m + j = i*4 + j  (since m=4)
        V row for B[j,k] = j*p + k = j*7 + k  (since p=7)
        W row for C[k,i] = k*n + i = k*2 + i  (since n=2)
        
        But we're working with sub-blocks, so we need to map local indices to global.
        For A sub-block at global row offset (in terms of A's column index j_global):
          local A[i][j_local] corresponds to global A[i][j_global + j_local]
          U index = i * m + (j_global + j_local)
        
        For B sub-block at global row offset (in terms of B's row index j_global):
          local B[j_local][k] corresponds to global B[j_global + j_local][k]
          V index = (j_global + j_local) * p + k
        
        For C sub-block at global col offset k_global:
          local C[k_local][i] corresponds to global C[k_global + k_local][i]
          W index = (k_global + k_local) * n + i
        """
        global col
        # Strassen 7 products
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        def U_idx(i_local, j_local):
            return i_local * m + (A_row_offset + j_local)
        
        def V_idx(j_local, k_local):
            return (B_row_offset + j_local) * p + k_local
        
        def W_idx(k_local, i_local):
            return (C_row_offset + k_local) * n + i_local
        
        # M1 = (a11+a22)(b11+b22)
        c = col
        U[U_idx(0,0)][c] = coeff * 1
        U[U_idx(1,1)][c] = coeff * 1
        V[V_idx(0,0)][c] = coeff * 1
        V[V_idx(1,1)][c] = coeff * 1
        W[W_idx(0,0)][c] = coeff * 1  # C11
        W[W_idx(1,1)][c] = coeff * 1  # C22
        col += 1
        
        # M2 = (a21+a22)b11
        c = col
        U[U_idx(1,0)][c] = coeff * 1
        U[U_idx(1,1)][c] = coeff * 1
        V[V_idx(0,0)][c] = coeff * 1
        W[W_idx(1,0)][c] = coeff * 1  # C21
        col += 1
        
        # M3 = a11(b21-b22)
        c = col
        U[U_idx(0,0)][c] = coeff * 1
        V[V_idx(1,0)][c] = coeff * 1
        V[V_idx(1,1)][c] = coeff * -1
        W[W_idx(0,1)][c] = coeff * 1  # C12
        W[W_idx(1,1)][c] = coeff * 1  # C22
        col += 1
        
        # M4 = a22(b12-b11)
        c = col
        U[U_idx(1,1)][c] = coeff * 1
        V[V_idx(0,1)][c] = coeff * 1
        V[V_idx(0,0)][c] = coeff * -1
        W[W_idx(0,0)][c] = coeff * 1  # C11
        W[W_idx(1,0)][c] = coeff * 1  # C21
        col += 1
        
        # M5 = (a11+a12)b22
        c = col
        U[U_idx(0,0)][c] = coeff * 1
        U[U_idx(0,1)][c] = coeff * 1
        V[V_idx(1,1)][c] = coeff * 1
        W[W_idx(0,0)][c] = coeff * -1  # C11
        W[W_idx(0,1)][c] = coeff * 1   # C12
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        c = col
        U[U_idx(1,0)][c] = coeff * 1
        U[U_idx(0,0)][c] = coeff * -1
        V[V_idx(0,0)][c] = coeff * 1
        V[V_idx(0,1)][c] = coeff * 1
        W[W_idx(1,1)][c] = coeff * 1   # C22
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        c = col
        U[U_idx(0,1)][c] = coeff * 1
        U[U_idx(1,1)][c] = coeff * -1
        V[V_idx(1,0)][c] = coeff * 1
        V[V_idx(1,1)][c] = coeff * 1
        W[W_idx(0,0)][c] = coeff * 1   # C11
        col += 1
        
        return 7
    
    def add_schoolbook_221(A_row_offset, B_row_offset, C_col_offset, coeff=1):
        """
        Add schoolbook 2x2 * 2x1 block (rank 4).
        A is 2x2, B is 2x1, C is 2x1.
        """
        global col
        def U_idx(i_local, j_local):
            return i_local * m + (A_row_offset + j_local)
        def V_idx(j_local):
            return (B_row_offset + j_local) * p + C_col_offset
        def W_idx(i_local):
            return C_col_offset * n + i_local
        
        # 4 products: C[i] = sum_j A[i,j]*B[j,0]
        for i in range(2):
            for j in range(2):
                c = col
                U[U_idx(i, j)][c] = coeff * 1
                V[V_idx(j)][c] = coeff * 1
                W[W_idx(i)][c] = coeff * 1
                col += 1
        return 4
    
    # Block 1: A1 (cols 0-1 of A) * B1 (rows 0-1 of B, cols 0-6 of B)
    # A1 is 2x2, at A col offset 0
    # B1 is 2x7, at B row offset 0
    
    # B1 split: cols 0-2 (2x3) and cols 3-6 (2x4)
    # 2x3 = 2x2 + 2x1
    
    # A1 * B1[cols 0-1]: Strassen 2x2*2x2, A col offset=0, B row offset=0, C col offset=0
    add_strassen_222(A_row_offset=0, B_row_offset=0, C_row_offset=0)
    
    # A1 * B1[col 2]: schoolbook 2x2*2x1, A col offset=0, B row offset=0, C col offset=2
    add_schoolbook_221(A_row_offset=0, B_row_offset=0, C_col_offset=2)
    
    # A1 * B1[cols 3-6]: two Strassen 2x2*2x2 blocks
    # cols 3-4: C col offset=3
    add_strassen_222(A_row_offset=0, B_row_offset=0, C_row_offset=3)
    # cols 5-6: C col offset=5
    add_strassen_222(A_row_offset=0, B_row_offset=0, C_row_offset=5)
    
    # Block 2: A2 (cols 2-3 of A) * B2 (rows 2-3 of B, cols 0-6 of B)
    # A2 is 2x2, at A col offset 2
    # B2 is 2x7, at B row offset 2
    
    # A2 * B2[cols 0-1]: Strassen 2x2*2x2, A col offset=2, B row offset=2, C col offset=0
    add_strassen_222(A_row_offset=2, B_row_offset=2, C_row_offset=0)
    
    # A2 * B2[col 2]: schoolbook 2x2*2x1, A col offset=2, B row offset=2, C col offset=2
    add_schoolbook_221(A_row_offset=2, B_row_offset=2, C_col_offset=2)
    
    # A2 * B2[cols 3-6]: two Strassen 2x2*2x2 blocks
    # cols 3-4: C col offset=3
    add_strassen_222(A_row_offset=2, B_row_offset=2, C_row_offset=3)
    # cols 5-6: C col offset=5
    add_strassen_222(A_row_offset=2, B_row_offset=2, C_row_offset=5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

