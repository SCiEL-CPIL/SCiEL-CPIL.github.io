"""Program for matrix multiplication 3x4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We will use a hybrid Strassen/schoolbook decomposition.
    # Split the 6 columns of B into 3 blocks of 2 columns each.
    # For each block, compute C_k = A * B_k (3x2 result).
    # C_k has 3 rows. 
    # Row 0 of C_k: 1x4 * 4x2 = 1x2. Use schoolbook: 4*2 = 8 mults.
    # Rows 1,2 of C_k: 2x4 * 4x2 = 2x2. Use Strassen: 7 mults.
    # Total per block: 8 + 7 = 15 mults.
    # Total rank: 3 * 15 = 45.
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for block in range(3):
        # B_k is columns 2*block and 2*block+1 of B
        # Indices for B_k in V: j*p + (2*block + l) for j in 0..3, l in 0..1
        
        # Part 1: Row 0 of A * B_k (1x4 * 4x2)
        # C[0, 2*block+l] = sum_j A[0, j] * B[j, 2*block+l]
        for j in range(4):
            for l in range(2):
                U[0 * m + j][col] = 1
                V[j * p + (2 * block + l)][col] = 1
                W[(2 * block + l) * n + 0][col] = 1
                col += 1
        
        # Part 2: Rows 1,2 of A * B_k (2x4 * 4x2)
        # Let A_bot = A[1:3, :] (2x4), B_k = B[:, 2*block:2*block+2] (4x2)
        # Split A_bot cols into 2+2: A_L (cols 0,1), A_R (cols 2,3)
        # Split B_k rows into 2+2: B_T (rows 0,1), B_B (rows 2,3)
        # C_bot = A_bot * B_k = A_L * B_T + A_R * B_B
        # A_L is 2x2, B_T is 2x2 -> 2x2 result (top half of C_bot)
        # A_R is 2x2, B_B is 2x2 -> 2x2 result (bottom half of C_bot)
        # We need to compute 2x2 * 2x2 for each, then add.
        # Strassen for 2x2 * 2x2:
        # Let X = A_L (2x2), Y = B_T (2x2). Result Z = X*Y.
        # Z[0,0] = M1 - M4 + M6 + M7
        # Z[0,1] = M3 + M4
        # Z[1,0] = M1 + M2
        # Z[1,1] = M5 - M2 + M3 + M7
        # where:
        # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        # M2 = (X[1,0]+X[1,1]) * Y[0,0]
        # M3 = X[0,0] * (Y[0,1]-Y[1,1])
        # M4 = X[1,1] * (Y[1,0]-Y[0,0])
        # M5 = (X[0,0]+X[0,1]) * Y[1,1]
        # M6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
        # M7 = (X[1,1]-X[0,1]) * (Y[1,0]+Y[1,1])
        
        # Here X = A_L:
        # X[0,0] = A[1,0], X[0,1] = A[1,1]
        # X[1,0] = A[2,0], X[1,1] = A[2,1]
        # Y = B_T:
        # Y[0,0] = B[0, 2*block], Y[0,1] = B[0, 2*block+1]
        # Y[1,0] = B[1, 2*block], Y[1,1] = B[1, 2*block+1]
        
        # Result Z1 = X*Y contributes to C_bot[0,:] and C_bot[1,:]? No.
        # A_L * B_T is (2x2)*(2x2) = 2x2.
        # The rows of A_L correspond to A rows 1 and 2.
        # The columns of B_T correspond to B rows 0 and 1.
        # So A_L * B_T computes the partial sum for C[1, :] and C[2, :] from B rows 0,1.
        # Specifically:
        # (A_L * B_T)[0, l] = A[1,0]B[0, 2b+l] + A[1,1]B[1, 2b+l] -> contributes to C[1, 2b+l]
        # (A_L * B_T)[1, l] = A[2,0]B[0, 2b+l] + A[2,1]B[1, 2b+l] -> contributes to C[2, 2b+l]
        
        # Similarly A_R * B_B:
        # X' = A_R: A[1,2], A[1,3]; A[2,2], A[2,3]
        # Y' = B_B: B[2, 2b], B[2, 2b+1]; B[3, 2b], B[3, 2b+1]
        # (A_R * B_B)[0, l] -> contributes to C[1, 2b+l]
        # (A_R * B_B)[1, l] -> contributes to C[2, 2b+l]
        
        # So C[1, 2b+l] = (A_L*B_T)[0,l] + (A_R*B_B)[0,l]
        #    C[2, 2b+l] = (A_L*B_T)[1,l] + (A_R*B_B)[1,l]
        
        # We will implement Strassen for the first 2x2 mult (A_L * B_T) and add to U,V,W.
        # Then Strassen for the second 2x2 mult (A_R * B_B) and add to U,V,W.
        
        def strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, base_col):
            # A is 2x2: A[A_rows[0], A_cols[0]], etc.
            # B is 2x2: B[B_rows[0], B_cols[0]], etc.
            # C is 2x2: C[C_rows[0], C_cols[0]], etc.
            # Adds 7 multiplications starting at base_col.
            # Returns new base_col.
            
            # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
            # X[0,0] = A[A_rows[0], A_cols[0]]
            # X[1,1] = A[A_rows[1], A_cols[1]]
            # Y[0,0] = B[B_rows[0], B_cols[0]]
            # Y[1,1] = B[B_rows[1], B_cols[1]]
            
            def add_term(a_row, a_col, b_row, b_col, c_row, c_col, coeff):
                nonlocal base_col
                U[a_row][base_col] += coeff
                V[b_row][base_col] += coeff
                W[c_row][base_col] += coeff
                base_col += 1
            
            # M1
            # (X00 + X11) * (Y00 + Y11)
            # Expand: X00*Y00 + X00*Y11 + X11*Y00 + X11*Y11
            # We want to add M1 to the appropriate places in C.
            # Z[0,0] += M1, Z[0,1] += 0, Z[1,0] += M1, Z[1,1] -= M1
            # Wait, Strassen formulas:
            # Z00 = M1 - M4 + M6 + M7
            # Z01 = M3 + M4
            # Z10 = M1 + M2
            # Z11 = M5 - M2 + M3 + M7
            
            # So M1 contributes: +1 to Z00, +1 to Z10.
            # M2 contributes: +1 to Z10, -1 to Z11.
            # M3 contributes: +1 to Z01, +1 to Z11.
            # M4 contributes: -1 to Z00, +1 to Z01.
            # M5 contributes: +1 to Z11.
            # M6 contributes: +1 to Z00.
            # M7 contributes: +1 to Z00, +1 to Z11.
            
            # Let's define the 4 corners of A, B, C.
            # A00 = A[A_rows[0], A_cols[0]]
            # A01 = A[A_rows[0], A_cols[1]]
            # A10 = A[A_rows[1], A_cols[0]]
            # A11 = A[A_rows[1], A_cols[1]]
            # B00 = B[B_rows[0], B_cols[0]]
            # B01 = B[B_rows[0], B_cols[1]]
            # B10 = B[B_rows[1], B_cols[0]]
            # B11 = B[B_rows[1], B_cols[1]]
            # C00 = C[C_rows[0], C_cols[0]]
            # C01 = C[C_rows[0], C_cols[1]]
            # C10 = C[C_rows[1], C_cols[0]]
            # C11 = C[C_rows[1], C_cols[1]]
            
            # M1 = (A00 + A11) * (B00 + B11)
            # We need to add 4 terms to U,V,W? No, we add ONE multiplication per M.
            # The multiplication is (A00 + A11) * (B00 + B11).
            # This requires adding A00 and A11 to the same U slot? No.
            # The tensor decomposition allows linear combinations in U, V, W.
            # So for M1, we set:
            # U[A00] += 1, U[A11] += 1
            # V[B00] += 1, V[B11] += 1
            # W[C00] += 1, W[C10] += 1
            # This creates the term (A00+A11)(B00+B11) for both C00 and C10.
            
            # Let's implement a helper to add a Strassen multiplication.
            # It takes the indices for A, B, C corners and adds the 7 mults.
            
            a00 = A_rows[0]*m + A_cols[0]
            a01 = A_rows[0]*m + A_cols[1]
            a10 = A_rows[1]*m + A_cols[0]
            a11 = A_rows[1]*m + A_cols[1]
            
            b00 = B_rows[0]*p + B_cols[0]
            b01 = B_rows[0]*p + B_cols[1]
            b10 = B_rows[1]*p + B_cols[0]
            b11 = B_rows[1]*p + B_cols[1]
            
            c00 = C_cols[0]*n + C_rows[0]
            c01 = C_cols[1]*n + C_rows[0]
            c10 = C_cols[0]*n + C_rows[1]
            c11 = C_cols[1]*n + C_rows[1]
            
            # M1 = (A00+A11)(B00+B11) -> C00+=1, C10+=1
            U[a00][base_col] += 1; U[a11][base_col] += 1
            V[b00][base_col] += 1; V[b11][base_col] += 1
            W[c00][base_col] += 1; W[c10][base_col] += 1
            base_col += 1
            
            # M2 = (A10+A11)B00 -> C10+=1, C11-=1
            U[a10][base_col] += 1; U[a11][base_col] += 1
            V[b00][base_col] += 1
            W[c10][base_col] += 1; W[c11][base_col] -= 1
            base_col += 1
            
            # M3 = A00(B01-B11) -> C01+=1, C11+=1
            U[a00][base_col] += 1
            V[b01][base_col] += 1; V[b11][base_col] -= 1
            W[c01][base_col] += 1; W[c11][base_col] += 1
            base_col += 1
            
            # M4 = A11(B10-B00) -> C00-=1, C01+=1
            U[a11][base_col] += 1
            V[b10][base_col] += 1; V[b00][base_col] -= 1
            W[c00][base_col] -= 1; W[c01][base_col] += 1
            base_col += 1
            
            # M5 = (A00+A01)B11 -> C11+=1
            U[a00][base_col] += 1; U[a01][base_col] += 1
            V[b11][base_col] += 1
            W[c11][base_col] += 1
            base_col += 1
            
            # M6 = (A10-A00)(B00+B01) -> C00+=1
            U[a10][base_col] += 1; U[a00][base_col] -= 1
            V[b00][base_col] += 1; V[b01][base_col] += 1
            W[c00][base_col] += 1
            base_col += 1
            
            # M7 = (A11-A01)(B10+B11) -> C00+=1, C11+=1
            U[a11][base_col] += 1; U[a01][base_col] -= 1
            V[b10][base_col] += 1; V[b11][base_col] += 1
            W[c00][base_col] += 1; W[c11][base_col] += 1
            base_col += 1
            
            return base_col
        
        # First Strassen: A_L (rows 1,2; cols 0,1) * B_T (rows 0,1; cols 0,1 of B_k)
        # A_rows = [1, 2], A_cols = [0, 1]
        # B_rows = [0, 1], B_cols = [2*block, 2*block+1]
        # C_rows = [1, 2], C_cols = [2*block, 2*block+1]
        col = strassen_2x2([1, 2], [0, 1], [0, 1], [2*block, 2*block+1], [1, 2], [2*block, 2*block+1], col)
        
        # Second Strassen: A_R (rows 1,2; cols 2,3) * B_B (rows 2,3; cols 0,1 of B_k)
        # A_rows = [1, 2], A_cols = [2, 3]
        # B_rows = [2, 3], B_cols = [2*block, 2*block+1]
        # C_rows = [1, 2], C_cols = [2*block, 2*block+1]
        col = strassen_2x2([1, 2], [2, 3], [2, 3], [2*block, 2*block+1], [1, 2], [2*block, 2*block+1], col)
        
    return {"U": U, "V": V, "W": W}
