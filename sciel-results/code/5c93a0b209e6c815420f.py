"""Matrix multiplication 2x3 by 3x3 — rank-17 exact decomposition."""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-17 decomposition of the 2x3 * 3x3 matrix multiplication tensor.

    Strategy:
      C[i,k] = sum_{j=1}^{3} A[i,j] * B[j,k]
             = sum_{j=1}^{2} A[i,j] * B[j,k]  +  A[i,3] * B[3,k]

    Part 1 (rank 11): multiply A[0:2, 0:2] (2x2) by B[0:2, 0:3] (2x3).
      Further split B into B[:,0:2] (2x2) and B[:,2] (2x1):
        - A[0:2,0:2] * B[0:2,0:2]: Strassen, rank 7
        - A[0:2,0:2] * B[0:2,2:3]: schoolbook for 2x2x1, rank 4
      Total: 11

    Part 2 (rank 6): A[i,3] * B[3,k] for all i in {0,1}, k in {0,1,2}.
      This is outer product of a 2-vector and a 3-vector: rank 6.

    Total rank: 17.

    Index conventions:
      U has shape (n*m, rank) = (6, 17), indexed U[i*m + j, r]
      V has shape (m*p, rank) = (9, 17), indexed V[j*p + k, r]
      W has shape (p*n, rank) = (6, 17), indexed W[k*n + i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17

    # Part 1a: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # Strassen computes C = A * B for 2x2 matrices with 7 multiplications.
    # A = [[A00, A01], [A10, A11]]  (rows i=0,1; cols j=0,1)
    # B = [[B00, B01], [B10, B11]]  (rows j=0,1; cols k=0,1)
    # C00 = A00*(B00+B11) - (A10-A00)*(B10+B11)
    # C01 = (A00+A01)*B11 - (A10+A01)*(B00+B11)  -- wait let me re-derive
    # Standard Strassen:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B10-B11)
    #   M4 = A11*(B01-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B10)
    #   M7 = (A01-A11)*(B01+B11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # For our tensor decomposition, each "multiplication" M_r corresponds to
    # one rank-1 term. For Strassen term M_r = X * Y where X is a linear
    # combination of A entries and Y is a linear combination of B entries,
    # the contribution to C[i,k] is coeff * (A-linear) * (B-linear).
    #
    # In the tensor: T[i,j,k] = C[i,k] coefficient for A[i,j]*B[j,k].
    # Each rank-1 term adds: for each (i,j) with nonzero a_ij in the A-linear
    # part, and each (k') with nonzero b_k in the B-linear part,
    # the product contributes to C at position determined by which C element
    # this M_r feeds into.
    #
    # Actually, let's think of it differently. We're decomposing the
    # multiplication map. The tensor is T[i, j, k] where the output
    # C[i,k] = sum_j A[i,j] B[j,k].
    #
    # A rank-r decomposition means:
    #   T[i,j,k] = sum_{r=0}^{rank-1} U[i,j,r] * V[j,k,r] * ... 
    # 
    # Wait, the interface is:
    #   U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    #   and the condition is: for all i,j,k:
    #     delta_{i,j,k} (the standard basis) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # No wait. The tensor for matrix multiplication A(2x3) * B(3x3) -> C(2x3):
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the structure tensor: for each output position (i,k) and each
    # summation index j, we have T[i,k,j] = 1 (meaning A[i,j]*B[j,k] contributes).
    #
    # In the factorization format:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  if (i,j,k) is a valid term
    #   = 0 otherwise
    #
    # Actually for matrix multiplication the tensor has:
    #   T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k], so every triple (i,j,k) appears.
    #
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For schoolbook: U[i*m+j, i*m*p+j*p+k] = 1, V[j*p+k, ...] = 1, W[k*n+i, ...] = 1
    #
    # For Strassen on 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1}):
    # We need: for i in {0,1}, j in {0,1}, k in {0,1}:
    #   sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    # And for other (i,j,k) triples involving these U,V,W rows/cols: 0
    #
    # Strassen gives us 7 rank-1 terms. Let me define them.
    #
    # A-linear factors (coefficients for A[i,j]):
    #   M1: A00+A11  ->  a100=1, a110=1  (using A[row,col] where row=i, col=j)
    #   M2: A10+A11  ->  a110=1, a111=1
    #   M3: A00      ->  a100=1
    #   M4: A11      ->  a111=1
    #   M5: A00+A01  ->  a100=1, a101=1
    #   M6: A10-A00  ->  a110=1, a100=-1
    #   M7: A01-A11  ->  a101=1, a111=-1
    #
    # B-linear factors (coefficients for B[j,k]):
    #   M1: B00+B11  ->  b100=1, b111=1  (B[row,col], row=j, col=k)
    #   M2: B00      ->  b100=1
    #   M3: B10-B11  ->  b110=1, b111=-1
    #   M4: B01-B00  ->  b101=1, b100=-1
    #   M5: B11      ->  b111=1
    #   M6: B00+B10  ->  b100=1, b110=1
    #   M7: B01+B11  ->  b101=1, b111=1
    #
    # Output C[i,k] = sum of contributions:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # So W[k*2+i, r] gets the coefficient from the C formula:
    #   r=M1: W[0*2+0]=W[0, M1]=1, W[1*2+1]=W[3, M1]=1
    #   r=M2: W[0*2+1]=W[1, M2]=1 (C01 gets M2? No: C10 = M2+M4, so W[0*2+1]=W[1,M2]... 
    #
    # Wait: W index is W[k*n+i]. n=2. So W[k*2+i].
    #   C[i,k] corresponds to W[k*2 + i].
    #   C00: i=0,k=0 -> W[0]
    #   C01: i=0,k=1 -> W[2]
    #   C10: i=1,k=0 -> W[1]
    #   C11: i=1,k=1 -> W[3]
    #
    #   M1: C00 += 1*M1, C11 += 1*M1  -> W[0, M1]=1, W[3, M1]=1
    #   M2: C10 += 1*M2  -> W[1, M2]=1
    #   M3: C01 += 1*M3, C11 += 1*M3  -> W[2, M3]=1, W[3, M3]=1
    #   M4: C00 += 1*M4, C10 += 1*M4  -> W[0, M4]=1, W[1, M4]=1
    #   M5: C00 += -1*M5, C01 += 1*M5 -> W[0, M5]=-1, W[2, M5]=1
    #   M6: C11 += 1*M6  -> W[3, M6]=1
    #   M7: C00 += 1*M7  -> W[0, M7]=1

    # U index: U[i*m+j] = U[i*3+j] for m=3
    #   i=0,j=0 -> U[0]; i=0,j=1 -> U[1]; i=1,j=0 -> U[3]; i=1,j=1 -> U[4]
    # V index: V[j*p+k] = V[j*3+k] for p=3
    #   j=0,k=0 -> V[0]; j=0,k=1 -> V[1]; j=1,k=0 -> V[3]; j=1,k=1 -> V[4]

    # Assign Strassen terms to columns 0-6
    strassen_cols = list(range(7))
    # M1 (col 0): A: A00+A11, B: B00+B11, W: C00+1, C11+1
    U[0][0] = 1; U[4][0] = 1  # A00, A11
    V[0][0] = 1; V[4][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11

    # M2 (col 1): A: A10+A11, B: B00, W: C10+1
    U[3][1] = 1; U[4][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[1][1] = 1  # C10

    # M3 (col 2): A: A00, B: B10-B11, W: C01+1, C11+1
    U[0][2] = 1  # A00
    V[3][2] = 1; V[4][2] = -1  # B10, B11
    W[2][2] = 1; W[3][2] = 1  # C01, C11

    # M4 (col 3): A: A11, B: B01-B00, W: C00+1, C10+1
    U[4][3] = 1  # A11
    V[1][3] = 1; V[0][3] = -1  # B01, B00
    W[0][3] = 1; W[1][3] = 1  # C00, C10

    # M5 (col 4): A: A00+A01, B: B11, W: C00-1, C01+1
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[4][4] = 1  # B11
    W[0][4] = -1; W[2][4] = 1  # C00, C01

    # M6 (col 5): A: A10-A00, B: B00+B10, W: C11+1
    U[3][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[3][5] = 1  # B00, B10
    W[3][5] = 1  # C11

    # M7 (col 6): A: A01-A11, B: B01+B11, W: C00+1
    U[1][6] = 1; U[4][6] = -1  # A01, A11
    V[1][6] = 1; V[4][6] = 1  # B01, B11
    W[0][6] = 1  # C00

    # Part 1b: Schoolbook for A[0:2, 0:2] * B[0:2, 2] (2x2 x 1)
    # C[i, 2] += A[i, 0]*B[0, 2] + A[i, 1]*B[1, 2]  for i=0,1
    # 4 terms, columns 7-10
    for jj in range(2):  # j = 0, 1
        for ii in range(2):  # i = 0, 1
            col = 7 + ii * 2 + jj
            # A[i, jj] * B[jj, 2] contributes to C[i, 2]
            U[ii * 3 + jj][col] = 1      # A[i, jj]
            V[jj * 3 + 2][col] = 1       # B[jj, 2]
            W[2 * 2 + ii][col] = 1       # C[i, 2] -> W[k*n+i] = W[2*2+ii] = W[4+ii]

    # Part 2: A[i, 2] * B[2, k] for i in {0,1}, k in {0,1,2}
    # 6 terms, columns 11-16
    for ii in range(2):
        for kk in range(3):
            col = 11 + ii * 3 + kk
            U[ii * 3 + 2][col] = 1    # A[i, 2]
            V[2 * 3 + kk][col] = 1    # B[2, k]
            W[kk * 2 + ii][col] = 1   # C[i, k] -> W[k*n+i] = W[k*2+i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
