"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition using 4 Strassen 2x2x2 blocks.
    
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x4) = [[B11, B12], [B21, B22]] where each is 2x2
    C (2x4) = [C1 | C2] where C1, C2 are 2x2
    
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    
    Use Strassen for each: rank 7 each, total 28.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)  
    # W shape: (p*n, rank) = (8, 28)
    # W index: W[k*n+i, r] where i is row of A/C, k is col of B/C
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Indexing:
    # U[i*m + j, r] where i=0,1 (row of A), j=0,1,2,3 (col of A)
    # V[j*p + k, r] where j=0,1,2,3 (row of B), k=0,1,2,3 (col of B)
    # W[k*n + i, r] where k=0,1,2,3 (col of C), i=0,1 (row of C)
    
    # A indices: A[i, j] -> U[i*4 + j, r]
    # B indices: B[j, k] -> V[j*4 + k, r]
    # C indices: C[i, k] -> W[k*2 + i, r]
    
    # Partition:
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C1 = C[0:2, 0:2], C2 = C[0:2, 2:4]
    
    # Strassen 7 products for X*Y (2x2 each):
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For C1 = A1*B11 + A2*B21:
    # This is not a single 2x2*2x2. We need to handle it differently.
    # C1[i,k] = sum_{j=0}^{1} A1[i,j]*B11[j,k] + sum_{j=0}^{1} A2[i,j]*B21[j,k]
    # 
    # Let me define for each Strassen block, the X and Y matrices.
    # For C1: X = A (2x2 rows, but we need to combine with B blocks)
    
    # Actually, let's think of it as:
    # C1 = A1*B11 + A2*B21
    # We can compute this using 7 Strassen products where:
    # The "left" factor involves A1 and A2, the "right" involves B11 and B21.
    # 
    # Define for C1:
    # X = A (2x2), but we need two different B's. 
    # 
    # Alternative: compute A1*B11 and A2*B21 separately using Strassen, 
    # but that's 7+7=14 for C1 alone.
    # 
    # Better: use the fact that C1[i,k] = sum_j (A1[i,j]*B11[j,k] + A2[i,j]*B21[j,k])
    # Define combined: for each (i,k), it's a bilinear form.
    # 
    # Let's use 7 products for the "combined" operation:
    # Think of it as multiplying a 2x4 matrix by a 4x2 matrix to get 2x2.
    # Left: [A1 | A2] (2x4), Right: [B11; B21] (4x2)
    # This is a 2x4 * 4x2 -> 2x2 multiplication.
    # 
    # For 2x4 * 4x2, we can use a block Strassen approach.
    # Split 2x4 into two 2x2 blocks, 4x2 into two 2x2 blocks.
    # Result = X1*Y1 + X2*Y2 where X1,X2 are 2x2, Y1,Y2 are 2x2.
    # 
    # Using Strassen on this sum: we can compute it with 7 products if we 
    # combine appropriately. Actually, X1*Y1 + X2*Y2 can be done with 
    # 7 products using a variant of Strassen.
    
    # Let me just implement it directly with 7 products per C block.
    
    # For C1 = [A1|A2] * [B11;B21] (2x4 * 4x2 -> 2x2):
    # Let X1=A1, X2=A2, Y1=B11, Y2=B21
    # M1 = (X1[0][0]+X1[1][1]) * (Y1[0][0]+Y2[1][1])  # This isn't right
    
    # I think the cleanest is to just do schoolbook for the 2x4*4x2 which is 16 products,
    # but use Strassen for the 2x2*2x2 subproducts.
    
    # Let me just do the straightforward 4 Strassen blocks:
    # C1 = A1*B11 + A2*B21: compute A1*B11 (7) + A2*B21 (7) = 14
    # C2 = A1*B12 + A2*B22: compute A1*B12 (7) + A2*B22 (7) = 14
    # Total: 28
    
    col = 0
    
    def strassen_block(X_rows, X_cols, Y_rows, Y_cols, C_rows, C_cols):
        """
        Add 7 Strassen products for X (2x2) * Y (2x2) -> C (2x2).
        X_rows, X_cols: row/col indices in A for the 2x2 block
        Y_rows, Y_cols: row/col indices in B for the 2x2 block  
        C_rows, C_cols: row/col indices in C for the 2x2 block
        Returns next column index.
        """
        nonlocal col
        # X[i,j] is at A[X_rows[i], X_cols[j]] -> U[X_rows[i]*4 + X_cols[j], col]
        # Y[i,j] is at B[Y_rows[i], Y_cols[j]] -> V[Y_rows[i]*4 + Y_cols[j], col]
        # C[i,j] is at C[C_rows[i], C_cols[j]] -> W[C_cols[j]*2 + C_rows[i], col]
        
        x00 = X_rows[0] * 4 + X_cols[0]
        x01 = X_rows[0] * 4 + X_cols[1]
        x10 = X_rows[1] * 4 + X_cols[0]
        x11 = X_rows[1] * 4 + X_cols[1]
        
        y00 = Y_rows[0] * 4 + Y_cols[0]
        y01 = Y_rows[0] * 4 + Y_cols[1]
        y10 = Y_rows[1] * 4 + Y_cols[0]
        y11 = Y_rows[1] * 4 + Y_cols[1]
        
        c00 = C_cols[0] * 2 + C_rows[0]
        c01 = C_cols[1] * 2 + C_rows[0]
        c10 = C_cols[0] * 2 + C_rows[1]
        c11 = C_cols[1] * 2 + C_rows[1]
        
        # M1 = (x00+x11)*(y00+y11)
        U[x00][col] += 1; U[x11][col] += 1
        V[y00][col] += 1; V[y11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (x10+x11)*y00
        U[x10][col] += 1; U[x11][col] += 1
        V[y00][col] += 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # M3 = x00*(y01-y11)
        U[x00][col] += 1
        V[y01][col] += 1; V[y11][col] += -1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = x11*(y10-y00)
        U[x11][col] += 1
        V[y10][col] += 1; V[y00][col] += -1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (x00+x01)*y11
        U[x00][col] += 1; U[x01][col] += 1
        V[y11][col] += 1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[x10][col] += 1; U[x00][col] += -1
        V[y00][col] += 1; V[y01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (x01-x11)*(y10+y11)
        U[x01][col] += 1; U[x11][col] += -1
        V[y10][col] += 1; V[y11][col] += 1
        W[c00][col] += 1; W[c11][col] += -1
        col += 1
        
        return col
    
    # C1 = A1*B11 + A2*B21
    # A1: rows 0,1 cols 0,1; A2: rows 0,1 cols 2,3
    # B11: rows 0,1 cols 0,1; B21: rows 2,3 cols 0,1
    # C1: rows 0,1 cols 0,1
    
    # A1*B11
    col = strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # A2*B21
    col = strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # C2 = A1*B12 + A2*B22
    # B12: rows 0,1 cols 2,3; B22: rows 2,3 cols 2,3
    # C2: rows 0,1 cols 2,3
    
    # A1*B12
    col = strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # A2*B22
    col = strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
