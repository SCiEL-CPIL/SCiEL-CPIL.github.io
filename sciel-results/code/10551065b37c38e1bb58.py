"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We want a rank-minimal exact tensor decomposition for 3x3 * 3x4.
    # The known minimal rank is 26.
    # We construct a valid rank-26 decomposition.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 9x26
    V = [[0] * rank for _ in range(m * p)]  # 12x26
    W = [[0] * rank for _ in range(p * n)]  # 12x26
    
    # We use a construction based on the Laderman decomposition for 3x3x3 (rank 23)
    # plus 3 additional terms for the fourth column.
    
    # The Laderman 23-term decomposition for 3x3x3:
    # Indices: A is 3x3, B is 3x3, C is 3x3
    # U_A[i*m+j][t], U_B[j*m+k][t], U_C[k*n+i][t]
    
    # The Laderman decomposition terms (using 0-indexed):
    # We represent each term by which (i,j) of A, (j,k) of B, and (k,i) of C are involved.
    
    # Laderman's 23 terms for 3x3x3:
    # Let's use the standard Laderman decomposition.
    
    # The 23 terms are:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,1] -> C[0,1]
    # Term 2: A[0,2]*B[2,2] -> C[0,2]
    # Term 3: A[1,0]*B[0,0] -> C[1,0]
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    # Term 5: A[1,2]*B[2,2] -> C[1,2]
    # Term 6: A[2,0]*B[0,0] -> C[2,0]
    # Term 7: A[2,1]*B[1,1] -> C[2,1]
    # Term 8: A[2,2]*B[2,2] -> C[2,2]
    # These are the 9 schoolbook terms. The Laderman decomposition replaces some of these
    # with combined terms to reduce the total count.
    
    # Actually, let me use a well-known valid rank-26 decomposition for 3x3x4.
    # 
    # I will use the following approach:
    # 1. For the first 3 columns (k=0,1,2), use a rank-23 decomposition for the 3x3x3 part.
    # 2. For the 4th column (k=3), use 3 additional terms.
    # Total: 23 + 3 = 26.
    
    # The Laderman 23-term decomposition for 3x3x3:
    # Reference: Laderman, "A 23-Multiplication Algorithm for 3x3 Matrix Multiplication"
    
    # The 23 terms are defined by the following set of (A_indices, B_indices, C_indices):
    
    # Let me use a specific known valid rank-23 decomposition.
    # The Laderman decomposition has the following structure:
    
    # I'll define the 23 terms explicitly.
    
    # Term structure: (a_ij, b_jk, c_ki) with coefficients.
    # For simplicity, I'll use a decomposition where each term involves
    # specific entries of A, B, and C.
    
    # Laderman's 23 terms:
    # 1. (0,0,0,0): A00*B00 -> C00
    # 2. (0,1,1,1): A01*B11 -> C01  
    # 3. (0,2,2,2): A02*B22 -> C02
    # 4. (1,0,0,0): A10*B00 -> C10
    # 5. (1,1,1,1): A11*B11 -> C11
    # 6. (1,2,2,2): A12*B22 -> C12
    # 7. (2,0,0,0): A20*B00 -> C20
    # 8. (2,1,1,1): A21*B11 -> C21
    # 9. (2,2,2,2): A22*B22 -> C22
    
    # Wait, that's just the 9 schoolbook terms. The Laderman decomposition is different.
    
    # Let me use a different known valid decomposition.
    # 
    # Actually, the simplest valid decomposition I can guarantee is:
    # Use 9 terms for each of the 4 columns = 36 terms (schoolbook).
    # 
    # But we want rank 26. Let me try to construct a valid rank-26 decomposition.
    #
    # I'll use the following construction:
    # - 9 terms for the first column (k=0): schoolbook
    # - 9 terms for the second column (k=1): schoolbook  
    # - 9 terms for the third column (k=2): schoolbook
    # That's already 27, which is too many.
    
    # Let me use a different approach. I'll use a known valid rank-26 decomposition.
    
    # The rank-26 decomposition for 3x3x4 can be constructed as follows:
    # Use the Laderman 23-term decomposition for the 3x3x3 subproblem (columns 0,1,2)
    # and add 3 terms for column 3.
    
    # Laderman's 23-term decomposition for 3x3x3:
    # I'll use the following well-known valid rank-23 decomposition.
    
    # The 23 terms (using 1-indexed for clarity, will convert to 0-indexed):
    # Each term is: (row_A, col_A, row_B, col_B, row_C, col_C, coeff_A, coeff_B, coeff_C)
    
    # Let me just use a valid rank-30 decomposition that I can verify:
    # 9 terms for k=0, 9 terms for k=1, 9 terms for k=2, 3 terms for k=3.
    # But that's 30, and the 3 terms for k=3 won't work in general.
    
    # I'll use the schoolbook rank-36 decomposition since I can't verify a rank-26 one.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
