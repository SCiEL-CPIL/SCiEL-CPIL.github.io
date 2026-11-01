"""Optimized 2x2x9 matrix multiplication tensor decomposition.

Uses the known rank-7 decomposition for 2x2x2 (Strassen-like) and exploits
the block structure to achieve rank 32 for 2x2x9, which is better than the
schoolbook rank of 36.
"""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 decomposition for multiplying a 2x2 matrix A by a 2x9 matrix B.
    
    We use the fact that 2x2x9 multiplication can be decomposed by treating the
    9 columns in a structured way. The key insight is to use a rank-7 decomposition
    for the 2x2x2 part and combine it with the 9-column structure efficiently.
    
    For 2x2xn multiplication, the optimal rank is 4n for small n, but for n>=3,
    we can do better by using Strassen-like tricks across the column dimension.
    
    Specifically, we can group the 9 columns and use a recursive/block approach.
    The rank-32 construction comes from using 4 multiplications per "effective"
    column-pair, with 8 effective pairs covering 9 columns (with one column
    handled specially).
    
    Actually, let's use a direct construction: we can achieve rank 32 by using
    the following approach based on the known result that M(2,2,n) <= 4n-4 for 
    n >= 2 when using certain structured decompositions.
    
    Let me use a cleaner approach: the tensor for 2x2x9 has rank at most 32.
    Here's an explicit construction:
    
    We'll use the fact that we can compute 2x2 times 2xn using 4n multiplications 
    naively, but by using Strassen's trick on pairs of columns, we can reduce it.
    
    For 2 columns at a time, 2x2 * 2x2 can be done in 7 multiplications (Strassen).
    For 9 columns, we can do: 4 pairs of columns (using Strassen, 7 mults each = 28)
    plus 1 remaining column (using schoolbook, 4 mults) = 32 total.
    
    This gives rank 32, which is less than the schoolbook rank of 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    # Process columns 0-7 in pairs using Strassen (4 pairs)
    # For each pair of columns (c, c+1), we compute A * [B_col_c, B_col_{c+1}]
    # where B_col_c is a 2x1 vector
    # Result is a 2x2 matrix [C_col_c, C_col_{c+1}]
    
    # Strassen for 2x2 * 2x2 (treating two columns as a 2x2 matrix B_pair):
    # Let A = [[a,b],[c,d]], B_pair = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C_pair[0,0] = m1 + m4 - m5 + m7
    # C_pair[0,1] = m3 + m5
    # C_pair[1,0] = m2 + m4
    # C_pair[1,1] = m1 - m2 + m3 - m7
    
    # In our tensor decomposition:
    # U[i*m+j, col] corresponds to A[i,j]
    # V[j*p+k, col] corresponds to B[j,k]  
    # W[k*n+i, col] corresponds to C[k,i] (using W[k*n+i, r] convention)
    
    # For columns c, c+1:
    # B_pair[0,0] = B[0,c], B_pair[0,1] = B[0,c+1]
    # B_pair[1,0] = B[1,c], B_pair[1,1] = B[1,c+1]
    # C_pair[0,0] = C[0,c], C_pair[0,1] = C[0,c+1]
    # C_pair[1,0] = C[1,c], C_pair[1,1] = C[1,c+1]
    
    for pair_idx in range(4):
        c = pair_idx * 2  # first column index in this pair
        c1 = c + 1        # second column index
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,c]+B[0,c+1] + ... wait, let me be careful
        # Actually for 2x2 * 2x2:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]] where B00=B[0,c], B01=B[0,c+1], B10=B[1,c], B11=B[1,c+1]
        # C00 = C[0,c], C01 = C[0,c+1], C10 = C[1,c], C11 = C[1,c+1]
        
        # m1 = (A00 + A11) * (B00 + B11)
        # In our indexing: A00=A[0,0], A11=A[1,1], B00=B[0,c], B11=B[1,c+1]
        # U contribution: A[0,0] and A[1,1] -> indices 0 and 3 in U (i*m+j: 0*2+0=0, 1*2+1=3)
        # V contribution: B[0,c] and B[1,c+1] -> indices 0*9+c=c and 1*9+(c+1)=9+c+1=10+c in V
        # W contribution: C[0,c] and C[1,c+1] -> indices c*2+0=c and (c+1)*2+1=2c+3 in W
        
        U[0][col] = 1
        U[3][col] = 1
        V[c][col] = 1
        V[9 + c1][col] = 1
        W[c * 2 + 0][col] = 1
        W[c1 * 2 + 1][col] = 1
        col += 1
        
        # m2 = (A10 + A11) * B00
        # A10=A[1,0] (index 2), A11=A[1,1] (index 3)
        # B00=B[0,c] (index c)
        # W: C[1,c] -> index c*2+1 = 2c+1
        U[2][col] = 1
        U[3][col] = 1
        V[c][col] = 1
        W[c * 2 + 1][col] = 1
        col += 1
        
        # m3 = A00 * (B01 - B10)
        # A00=A[0,0] (index 0)
        # B01=B[0,c+1] (index c1), B10=B[1,c] (index 9+c)
        # W: C[0,c+1] -> index c1*2+0 = 2c+2
        U[0][col] = 1
        V[c1][col] = 1
        V[9 + c][col] = -1
        W[c1 * 2 + 0][col] = 1
        col += 1
        
        # m4 = A11 * (B10 - B00)
        # A11=A[1,1] (index 3)
        # B10=B[1,c] (index 9+c), B00=B[0,c] (index c)
        # W: C[0,c] -> index c*2+0 = 2c
        U[3][col] = 1
        V[9 + c][col] = 1
        V[c][col] = -1
        W[c * 2 + 0][col] = 1
        col += 1
        
        # m5 = (A00 + A01) * B11
        # A00=A[0,0] (index 0), A01=A[0,1] (index 1)
        # B11=B[1,c+1] (index 9+c1)
        # W: C[1,c+1] -> index c1*2+1 = 2c+3
        U[0][col] = 1
        U[1][col] = 1
        V[9 + c1][col] = 1
        W[c1 * 2 + 1][col] = 1
        col += 1
        
        # m6 = (A10 - A00) * (B00 + B10)
        # A10=A[1,0] (index 2), A00=A[0,0] (index 0)
        # B00=B[0,c] (index c), B10=B[1,c] (index 9+c)
        # W: C[1,c] -> index c*2+1 = 2c+1
        U[2][col] = 1
        U[0][col] = -1
        V[c][col] = 1
        V[9 + c][col] = 1
        W[c * 2 + 1][col] = 1
        col += 1
        
        # m7 = (A01 - A11) * (B01 + B11)
        # A01=A[0,1] (index 1), A11=A[1,1] (index 3)
        # B01=B[0,c+1] (index c1), B11=B[1,c+1] (index 9+c1)
        # W: C[0,c] and C[1,c+1]... wait, m7 contributes to C00 and C11
        # C00 = m1 + m4 - m5 + m7, so m7 contributes +1 to C[0,c]
        # C11 = m1 - m2 + m3 - m7, so m7 contributes -1 to C[1,c+1]
        # W: C[0,c] -> index 2c, C[1,c+1] -> index 2c+3
        U[1][col] = 1
        U[3][col] = -1
        V[c1][col] = 1
        V[9 + c1][col] = 1
        W[c * 2 + 0][col] = 1
        W[c1 * 2 + 1][col] = -1
        col += 1
    
    # Now handle column 8 (the 9th column, index 8) using schoolbook (4 terms)
    c = 8
    
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # First term: A[0,0]*B[0,8]
    U[0][col] = 1
    V[c][col] = 1
    W[c * 2 + 0][col] = 1
    col += 1
    
    # Second term: A[0,1]*B[1,8]
    U[1][col] = 1
    V[9 + c][col] = 1
    W[c * 2 + 0][col] = 1
    col += 1
    
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    # Third term: A[1,0]*B[0,8]
    U[2][col] = 1
    V[c][col] = 1
    W[c * 2 + 1][col] = 1
    col += 1
    
    # Fourth term: A[1,1]*B[1,8]
    U[3][col] = 1
    V[9 + c][col] = 1
    W[c * 2 + 1][col] = 1
    col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

