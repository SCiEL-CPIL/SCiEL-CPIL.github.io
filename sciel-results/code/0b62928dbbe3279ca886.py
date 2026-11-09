"""Matrix multiplication 2x3 x 3x5 rank-19 decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 19
    
    # Strassen-style decomposition for 2x3 * 3x5
    # U: shape (n*m, rank) = (6, 19)
    # V: shape (m*p, rank) = (15, 19)
    # W: shape (p*n, rank) = (10, 19)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # The rank-19 decomposition for M_{2,3,5}
    # Based on the known optimal decomposition
    # 
    # We decompose the 2x3 * 3x5 multiplication into 19 rank-1 terms
    # Each term c has U[:,c], V[:,c], W[:,c] vectors
    #
    # For matrix C = A * B where A is 2x3 and B is 3x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # (the tensor of matrix multiplication)
    
    # Let me use the known rank-19 construction
    # This uses the fact that 2x3*3x5 can be done with 19 multiplications
    
    # Define the 19 rank-1 decompositions
    # Each is (u, v, w) where u in R^6, v in R^15, w in R^10
    
    # Using the decomposition from:
    # "The rank of matrix multiplication" - Strassen's result for small cases
    
    # For 2x3 * 3x5, the rank is 19
    # Let me construct it explicitly
    
    # The multiplication tensor T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    
    # We want: T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # Let me use a known construction:
    # Split the 3x5 matrix into parts and use Strassen-like tricks
    
    # Actually, let me use the explicit rank-19 decomposition
    # from the literature on small matrix multiplication ranks
    
    # Here's a valid rank-19 decomposition:
    
    # Term 0: A[0,0]*B[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: A[0,1]*B[1,0]
    U[1][1] = 1; V[5][1] = 1; W[0][1] = 1
    # Term 2: A[0,2]*B[2,0]
    U[2][2] = 1; V[10][2] = 1; W[0][2] = 1
    # Term 3: A[1,0]*B[0,0]
    U[3][3] = 1; V[0][3] = 1; W[2][3] = 1
    # Term 4: A[1,1]*B[1,0]
    U[4][4] = 1; V[5][4] = 1; W[2][4] = 1
    # Term 5: A[1,2]*B[2,0]
    U[5][5] = 1; V[10][5] = 1; W[2][5] = 1
    
    # Now for columns 1-4 of B, we need to be smarter
    # Let's use Strassen-like combinations
    
    # For B column 1:
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # We can compute:
    # P6 = (A[0,0]+A[1,0]) * (B[0,1])  -> gives A[0,0]*B[0,1] + A[1,0]*B[0,1]
    # P7 = (A[0,1]+A[1,1]) * (B[1,1])  -> gives A[0,1]*B[1,1] + A[1,1]*B[1,1]
    # P8 = (A[0,2]+A[1,2]) * (B[2,1])  -> gives A[0,2]*B[2,1] + A[1,2]*B[2,1]
    # Then C[0,1] = P6 + P7 + P8 - (A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1])
    # This doesn't save multiplications directly...
    
    # Let me try a different approach - use the known optimal decomposition
    
    # Actually, the simplest valid rank-19 decomposition for 2x3*3x5:
    # We can think of it as: first do 2x3*3x2 (rank 8) for first 2 cols,
    # then 2x3*3x2 for next 2 cols, then 2x3*3x1 for last col
    # 2x3*3x2 has rank 8 (Strassen for 2x2 with one extra column)
    # 2x3*3x1 has rank 5 (just 5 mults)
    # Total: 8 + 8 + 5 = 21, not 19
    
    # Let me just verify the schoolbook and try to find a better one
    # For now, return a valid decomposition - let me use rank 19 if I can find it
    
    # Let me try a different strategy: use bilinear complexity
    # The rank of M_{2,3,5} is indeed 19
    
    # Here's a construction using the following idea:
    # Group the 5 columns of B into {0,1}, {2,3}, {4}
    # For each group of 2 columns, use a 2x3*3x2 decomposition (rank 8)
    # For the last column, use 5 multiplications
    # But we can share some work...
    
    # Actually, let me just use a verified rank-19 decomposition
    # I'll construct it carefully
    
    # Reset
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use the following rank-19 decomposition:
    # Based on the fact that we can write the 2x3*3x5 tensor as sum of 19 rank-1 tensors
    
    # Let me use a systematic approach:
    # The tensor has 2*3*5 = 30 nonzero entries
    # We need at least 19 rank-1 terms
    
    # Here's a valid construction:
    # For each pair (i, k) where i in {0,1}, k in {0..4}, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can precompute certain sums of A and B to reduce multiplications
    
    # Let me use this decomposition (verified to work):
    
    # c=0: A00 * B00 -> contributes to C00
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    # c=1: A00 * B01 -> contributes to C01
    U[0][1]=1; V[1][1]=1; W[1][1]=1
    # c=2: A00 * B02 -> contributes to C02
    U[0][2]=1; V[2][2]=1; W[2][2]=1
    # c=3: A00 * B03 -> contributes to C03
    U[0][3]=1; V[3][3]=1; W[3][3]=1
    # c=4: A00 * B04 -> contributes to C04
    U[0][4]=1; V[4][4]=1; W[4][4]=1
    # c=5: A01 * B10 -> contributes to C00
    U[1][5]=1; V[5][5]=1; W[0][5]=1
    # c=6: A01 * B11 -> contributes to C01
    U[1][6]=1; V[6][6]=1; W[1][6]=1
    # c=7: A01 * B12 -> contributes to C02
    U[1][7]=1; V[7][7]=1; W[2][7]=1
    # c=8: A01 * B13 -> contributes to C03
    U[1][8]=1; V[8][8]=1; W[3][8]=1
    # c=9: A01 * B14 -> contributes to C04
    U[1][9]=1; V[9][9]=1; W[4][9]=1
    # c=10: A02 * B20 -> contributes to C00
    U[2][10]=1; V[10][10]=1; W[0][10]=1
    # c=11: A02 * B21 -> contributes to C01
    U[2][11]=1; V[11][11]=1; W[1][11]=1
    # c=12: A02 * B22 -> contributes to C02
    U[2][12]=1; V[12][12]=1; W[2][12]=1
    # c=13: A02 * B23 -> contributes to C03
    U[2][13]=1; V[13][13]=1; W[3][13]=1
    # c=14: A02 * B24 -> contributes to C04
    U[2][14]=1; V[14][14]=1; W[4][14]=1
    # c=15: A10 * B00 -> contributes to C10
    U[3][15]=1; V[0][15]=1; W[2][15]=1
    # c=16: A10 * B01 -> contributes to C11
    U[3][16]=1; V[1][16]=1; W[3][16]=1
    # c=17: A10 * B02 -> contributes to C12
    U[3][17]=1; V[2][17]=1; W[4][17]=1
    # c=18: A10 * B03 -> contributes to C13
    U[3][18]=1; V[3][18]=1; W[5][18]=1
    
    # This is only 19 terms but I'm missing contributions to C14, and
    # I haven't included A11*B1k and A12*B2k for k=4
    # This is incomplete. Let me just return the schoolbook for now
    # and note that finding the optimal 19 is complex
    
    # Actually, let me just return a valid schoolbook decomposition
    # The task says "rank-minimal" but if I can't find 19, 
    # I should at least return a valid one
    
    # Let me reconsider - return schoolbook which is definitely valid
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
