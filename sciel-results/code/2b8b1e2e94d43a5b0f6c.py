import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: recursive Strassen for 4x4x4 (rank 56) + schoolbook for 4x4x1 (rank 16) = rank 72
    # But we can do better: 4x4x2 via Strassen blocks (rank 28) + 4x4x3 via (4x4x2 rank 28 + 4x4x1 rank 16) = 44
    # Total: 28 + 44 = 72
    
    # Actually, let's use the cleanest approach:
    # 4x4x4 via 8 Strassen 2x2x2 sub-problems = rank 56
    # 4x4x1 via schoolbook = rank 16
    # Total = 72
    
    rank_total = 72
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank_total for _ in range(m * p)]  # 20 x 72
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x 72
    
    col = 0
    
    # Part 1: 4x4x4 part (k=0,1,2,3), rank 56
    # Block structure: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each block is 2x2. C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each 2x2x2 product uses Strassen rank 7.
    # 8 products total, each rank 7 = 56
    
    def add_strassen_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset, col_start):
        """Add a Strassen 2x2x2 decomposition starting at col_start.
        a_row_offset: starting row index in A (0, 2)
        b_col_offset: starting col index in B (0, 2)
        c_row_offset: starting row index in C (0, 2)
        c_col_offset: starting col index in C (0, 2)
        Returns number of columns used (7).
        """
        # A block rows: a_row_offset, a_row_offset+1
        # A block cols: 0,1,2,3 (full width since A is 4x4, but for the block multiplication
        #   we consider the 2x2 block of A times 2x2 block of B)
        # Actually, for 4x4x4 with 2x2 blocks:
        #   A11 is rows 0-1, cols 0-1 of A
        #   A12 is rows 0-1, cols 2-3 of A
        #   A21 is rows 2-3, cols 0-1 of A
        #   A22 is rows 2-3, cols 2-3 of A
        #   B11 is rows 0-1, cols 0-1 of B
        #   B12 is rows 0-1, cols 2-3 of B
        #   B21 is rows 2-3, cols 0-1 of B
        #   B22 is rows 2-3, cols 2-3 of B
        #   C11 is rows 0-1, cols 0-1 of C
        #   etc.
        
        # For a specific 2x2x2 sub-problem:
        #   A_sub is 2x2 (rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1)
        #   B_sub is 2x2 (rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1)
        #   C_sub is 2x2 (rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1)
        
        # Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A10)*B11
        # M6 = (A01-A11)*(B10+B11)
        # M7 = (A00-A10)*(B00+B01)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A_sub indices: (a_row_offset + dr, a_col_offset + dc) for dr,dc in {0,1}
        # B_sub indices: (b_row_offset + dr, b_col_offset + dc) for dr,dc in {0,1}
        # C_sub indices: (c_row_offset + dr, c_col_offset + dc) for dr,dc in {0,1}
        
        a_col_offset = a_row_offset  # For 4x4x4, A's col offset matches row offset for block structure
        # Wait, that's not right. Let me pass a_col_offset explicitly.
        
        return col_start
    
    # Let me restructure. I'll pass all offsets explicitly.
    
    def strassen_2x2x2(a_r, a_c, b_r, b_c, c_r, c_c, col_start):
        """Add Strassen 2x2x2 decomposition.
        A sub-block: rows a_r..a_r+1, cols a_c..a_c+1
        B sub-block: rows b_r..b_r+1, cols b_c..b_c+1
        C sub-block: rows c_r..c_r+1, cols c_c..c_c+1
        Returns new column index.
        """
        col = col_start
        
        # Define helper to add a term
        def add_term(a_idx, b_idx, c_idx, coeff=1):
            nonlocal col
            U[a_idx][col] += coeff
            V[b_idx][col] += coeff
            W[c_idx][col] += coeff
            col += 1
        
        # Strassen M1-M7
        # M1 = (A00+A11)*(B00+B11)
        # A00 = (a_r, a_c), A11 = (a_r+1, a_c+1)
        # B00 = (b_r, b_c), B11 = (b_r+1, b_c+1)
        # C contributions: C00 += M1, C11 += M1
        # We need to represent M1 as sum of products.
        # M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # But that's 4 terms, not 1. Strassen uses 7 multiplications total for the 2x2x2.
        
        # The 7 Strassen multiplications are:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A10)*B11
        # M6 = (A01-A11)*(B10+B11)
        # M7 = (A00-A10)*(B00+B01)
        
        # Each M_i is a single "multiplication" in the tensor sense, meaning
        # we need to represent each as a product of sums of A entries times sums of B entries.
        # In the tensor decomposition, each rank-1 term is U[i,j]*V[j,k]*W[k,i].
        # For a 2x2x2 sub-problem, each of the 7 Strassen terms corresponds to a rank-7
        # decomposition of the 2x2x2 multiplication tensor.
        
        # Actually, the standard Strassen 2x2x2 decomposition has rank 7, meaning
        # 7 rank-1 terms. Let me list them explicitly.
        
        # The 2x2x2 multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
        # Strassen gives 7 terms. Let me use the standard formulation:
        
        # Term 1: (A00+A11)*(B00+B11)
        #   U: A00 += 1, A11 += 1  => U[a_r*4+a_c] += 1, U[(a_r+1)*4+(a_c+1)] += 1
        #   V: B00 += 1, B11 += 1  => V[b_r*5+b_c] += 1, V[(b_r+1)*5+(b_c+1)] += 1
        #   W: C00 += 1, C11 += 1  => W[c_c*4+c_r] += 1, W[(c_c+1)*4+(c_r+1)] += 1
        #   Wait, W is indexed as W[k*n+i, r] where k is column of C, i is row of C.
        #   So W[k*n + i] for C[i,k].
        #   C00: i=0, k=0 => W[0*4+0] = W[0]
        #   C11: i=1, k=1 => W[1*4+1] = W[5]
        
        # Let me be very careful with indexing.
        # U[i*m+j] where i is row of A, j is col of A. m=4.
        # V[j*p+k] where j is row of B, k is col of B. p=5.
        # W[k*n+i] where k is col of C, i is row of C. n=4.
        
        # For the 2x2x2 sub-problem at:
        #   A: rows a_r, a_r+1; cols a_c, a_c+1
        #   B: rows b_r, b_r+1; cols b_c, b_c+1
        #   C: rows c_r, c_r+1; cols c_c, c_c+1
        
        # Local indices:
        #   A00 = A[a_r, a_c], A01 = A[a_r, a_c+1], A10 = A[a_r+1, a_c], A11 = A[a_r+1, a_c+1]
        #   B00 = B[b_r, b_c], B01 = B[b_r, b_c+1], B10 = B[b_r+1, b_c], B11 = B[b_r+1, b_c+1]
        #   C00 = C[c_r, c_c], C01 = C[c_r, c_c+1], C10 = C[c_r+1, c_c], C11 = C[c_r+1, c_c+1]
        
        # U indices:
        #   U_A00 = U[a_r * m + a_c]
        #   U_A01 = U[a_r * m + a_c + 1]
        #   U_A10 = U[(a_r+1) * m + a_c]
        #   U_A11 = U[(a_r+1) * m + a_c + 1]
        
        # V indices:
        #   V_B00 = V[b_r * p + b_c]
        #   V_B01 = V[b_r * p + b_c + 1]
        #   V_B10 = V[(b_r+1) * p + b_c]
        #   V_B11 = V[(b_r+1) * p + b_c + 1]
        
        # W indices:
        #   W_C00 = W[c_c * n + c_r]
        #   W_C01 = W[(c_c+1) * n + c_r]
        #   W_C10 = W[c_c * n + c_r + 1]
        #   W_C11 = W[(c_c+1) * n + c_r + 1]
        
        # Strassen 7 terms:
        
        # M1 = (A00+A11)*(B00+B11)
        #   Contributes to: C00 += M1, C11 += M1
        U[a_r * m + a_c][col] += 1
        U[(a_r+1) * m + (a_c+1)][col] += 1
        V[b_r * p + b_c][col] += 1
        V[(b_r+1) * p + (b_c+1)][col] += 1
        W[c_c * n + c_r][col] += 1
        W[(c_c+1) * n + (c_r+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B00
        #   Contributes to: C10 += M2, C11 -= M2
        U[(a_r+1) * m + a_c][col] += 1
        U[(a_r+1) * m + (a_c+1)][col] += 1
        V[b_r * p + b_c][col] += 1
        W[c_c * n + (c_r+1)][col] += 1
        W[(c_c+1) * n + (c_r+1)][col] += -1
        col += 1
        
        # M3 = A00*(B01-B11)
        #   Contributes to: C01 += M3, C11 += M3
        U[a_r * m + a_c][col] += 1
        V[b_r * p + (b_c+1)][col] += 1
        V[(b_r+1) * p + (b_c+1)][col] += -1
        W[(c_c+1) * n + c_r][col] += 1
        W[(c_c+1) * n + (c_r+1)][col] += 1
        col += 1
        
        # M4 = A11*(B10-B00)
        #   Contributes to: C00 += M4, C10 += M4
        U[(a_r+1) * m + (a_c+1)][col] += 1
        V[(b_r+1) * p + b_c][col] += 1
        V[b_r * p + b_c][col] += -1
        W[c_c * n + c_r][col] += 1
        W[c_c * n + (c_r+1)][col] += 1
        col += 1
        
        # M5 = (A00+A10)*B11
        #   Contributes to: C00 -= M5, C01 += M5
        U[a_r * m + a_c][col] += 1
        U[(a_r+1) * m + a_c][col] += 1
        V[(b_r+1) * p + (b_c+1)][col] += 1
        W[c_c * n + c_r][col] += -1
        W[(c_c+1) * n + c_r][col] += 1
        col += 1
        
        # M6 = (A01-A11)*(B10+B11)
        #   Contributes to: C11 += M6
        U[a_r * m + (a_c+1)][col] += 1
        U[(a_r+1) * m + (a_c+1)][col] += -1
        V[(b_r+1) * p + b_c][col] += 1
        V[(b_r+1) * p + (b_c+1)][col] += 1
        W[(c_c+1) * n + (c_r+1)][col] += 1
        col += 1
        
        # M7 = (A00-A10)*(B00+B01)
        #   Contributes to: C00 += M7
        U[a_r * m + a_c][col] += 1
        U[(a_r+1) * m + a_c][col] += -1
        V[b_r * p + b_c][col] += 1
        V[b_r * p + (b_c+1)][col] += 1
        W[c_c * n + c_r][col] += 1
        col += 1
        
        return col
    
    # Now apply Strassen to the 8 2x2x2 sub-problems of the 4x4x4 part.
    # Block structure:
    #   A11: a_r=0, a_c=0
    #   A12: a_r=0, a_c=2
    #   A21: a_r=2, a_c=0
    #   A22: a_r=2, a_c=2
    #   B11: b_r=0, b_c=0
    #   B12: b_r=0, b_c=2
    #   B21: b_r=2, b_c=0
    #   B22: b_r=2, b_c=2
    #   C11: c_r=0, c_c=0
    #   C12: c_r=0, c_c=2
    #   C21: c_r=2, c_c=0
    #   C22: c_r=2, c_c=2
    
    # C11 = A11*B11 + A12*B21
    #   Sub-problem 1: A11*B11 -> C11
    col = strassen_2x2x2(0, 0, 0, 0, 0, 0, col)
    #   Sub-problem 2: A12*B21 -> C11
    col = strassen_2x2x2(0, 2, 2, 0, 0, 0, col)
    
    # C12 = A11*B12 + A12*B22
    #   Sub-problem 3: A11*B12 -> C12
    col = strassen_2x2x2(0, 0, 0, 2, 0, 2, col)
    #   Sub-problem 4: A12*B22 -> C12
    col = strassen_2x2x2(0, 2, 2, 2, 0, 2, col)
    
    # C21 = A21*B11 + A22*B21
    #   Sub-problem 5: A21*B11 -> C21
    col = strassen_2x2x2(2, 0, 0, 0, 2, 0, col)
    #   Sub-problem 6: A22*B21 -> C21
    col = strassen_2x2x2(2, 2, 2, 0, 2, 0, col)
    
    # C22 = A21*B12 + A22*B22
    #   Sub-problem 7: A21*B12 -> C22
    col = strassen_2x2x2(2, 0, 0, 2, 2, 2, col)
    #   Sub-problem 8: A22*B22 -> C22
    col = strassen_2x2x2(2, 2, 2, 2, 2, 2, col)
    
    # Part 2: 4x4x1 part (k=4), rank 16 (schoolbook)
    # For k=4: C[i,4] = sum_j A[i,j]*B[j,4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    assert col == rank_total, f"Expected {rank_total} columns, got {col}"
    
    # Verify the decomposition
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank_total):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err += (val - T[i,j,k])**2
    
    if err < 1e-8:
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, fall back to schoolbook
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
