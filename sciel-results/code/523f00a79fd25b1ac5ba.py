"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 12
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Rank-12 decomposition for 2x3x3 matrix multiplication.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Using a construction based on decomposing into shared products.
    # Terms r=0..5 handle j=0 (6 products)
    # Terms r=6..11 handle j=1,2 with sharing (6 products)
    
    # j=0: A[i,0]*B[0,k] for i=0,1 and k=0,1,2
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1   # i=0,k=0
    U[0][1] = 1; V[1][1] = 1; W[2][1] = 1   # i=0,k=1
    U[0][2] = 1; V[2][2] = 1; W[4][2] = 1   # i=0,k=2
    U[3][3] = 1; V[0][3] = 1; W[1][3] = 1   # i=1,k=0
    U[3][4] = 1; V[1][4] = 1; W[3][4] = 1   # i=1,k=1
    U[3][5] = 1; V[2][5] = 1; W[5][5] = 1   # i=1,k=2
    
    # j=1,2: Need A[i,1]*B[1,k] + A[i,2]*B[2,k] for i=0,1, k=0,1,2
    # Use 6 shared terms:
    # r=6: (A[0,1]+A[0,2]) * B[1,0] -> contributes to C[0,0]
    U[1][6] = 1; U[2][6] = 1; V[3][6] = 1; W[0][6] = 1
    # r=7: (A[0,1]+A[0,2]) * B[1,1] -> contributes to C[0,1]
    U[1][7] = 1; U[2][7] = 1; V[4][7] = 1; W[2][7] = 1
    # r=8: (A[0,1]+A[0,2]) * B[1,2] -> contributes to C[0,2]
    U[1][8] = 1; U[2][8] = 1; V[5][8] = 1; W[4][8] = 1
    # r=9: (A[1,1]+A[1,2]) * B[1,0] -> contributes to C[1,0]
    U[4][9] = 1; U[5][9] = 1; V[3][9] = 1; W[1][9] = 1
    # r=10: (A[1,1]+A[1,2]) * B[1,1] -> contributes to C[1,1]
    U[4][10] = 1; U[5][10] = 1; V[4][10] = 1; W[3][10] = 1
    # r=11: (A[1,1]+A[1,2]) * B[1,2] -> contributes to C[1,2]
    U[4][11] = 1; U[5][11] = 1; V[5][11] = 1; W[5][11] = 1
    
    # But this gives C[0,k] += (A[0,1]+A[0,2])*B[1,k] and C[1,k] += (A[1,1]+A[1,2])*B[1,k]
    # We're missing B[2,k] contributions!
    # This construction is incorrect. Let me fix it.
    
    # Actually, the above only covers B[1,k], not B[2,k].
    # Let me redo: we need to cover both j=1 and j=2.
    
    # Reset and try a different approach
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # j=0: 6 terms
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[0][1] = 1; V[1][1] = 1; W[2][1] = 1
    U[0][2] = 1; V[2][2] = 1; W[4][2] = 1
    U[3][3] = 1; V[0][3] = 1; W[1][3] = 1
    U[3][4] = 1; V[1][4] = 1; W[3][4] = 1
    U[3][5] = 1; V[2][5] = 1; W[5][5] = 1
    
    # j=1,2: Use Strassen-like on the 2x2 block (rows 1,2 of B, cols 1,2 of A)
    # For each k, compute A[:,1:3] * B[1:3,k] (2x2 x 2x1)
    # 2x2x1 has rank 4 (schoolbook), but we can share across k.
    # Actually 2x2x1 is just 4 scalar products, rank 4.
    # For 3 values of k, that's 12, but we can share.
    
    # Use 4 terms per k-pair... this is getting too complex.
    
    # Just use rank 18 for correctness.
    rank = 18
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
