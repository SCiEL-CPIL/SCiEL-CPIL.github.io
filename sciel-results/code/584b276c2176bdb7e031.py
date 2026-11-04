# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 5 decomposition for 2x3 x 3x1
    # U: 2x3 x 5
    # V: 3x1 x 5
    # W: 1x2 x 5 (or 2x1 depending on convention)
    
    # Laderman's algorithm for 2x3?
    # Actually, let's use a known rank 5 decomposition for 2x3x1
    # c0 = a00*b0 + a01*b1 + a02*b2
    # c1 = a10*b0 + a11*b1 + a12*b2
    
    # We can use 5 multiplications.
    # m1 = (a00+a10)*(b0)
    # m2 = (a01+a11)*(b1)
    # m3 = (a02+a12)*(b2)
    # m4 = (a00+a01+a10+a11)*(b0+b1) ? No.
    
    # Let's just use the schoolbook for the 2x3x1 part (rank 3) and see if we can reduce.
    # Schoolbook for 2x3x1 is rank 3 (one for each j).
    # c0 = a00*b0 + a01*b1 + a02*b2
    # c1 = a10*b0 + a11*b1 + a12*b2
    # Rank 3:
    # r0: U[0,0]=1, V[0]=1, W[0,0]=1; U[1,0]=1, V[0]=1, W[0,1]=1
    # r1: U[0,1]=1, V[1]=1, W[1,0]=1; U[1,1]=1, V[1]=1, W[1,1]=1
    # r2: U[0,2]=1, V[2]=1, W[2,0]=1; U[1,2]=1, V[2]=1, W[2,1]=1
    
    # This gives rank 3 for each column.
    # Total rank 3 * 7 = 21.
    # This is better than 42.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            # U[i*m+j, col]
            # V[j*p+k, col]
            # W[k*n+i, col]
            
            # For i=0
            U[0*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 0][col] = 1
            
            # For i=1
            # We need a separate term for i=1?
            # The sum is over r.
            # We want sum_r U[0,j,r] V[j,k,r] W[k,0,r] = 1
            # and sum_r U[1,j,r] V[j,k,r] W[k,1,r] = 1
            
            # If we use the same r for i=0 and i=1, we need:
            # U[0,j,r] V[j,k,r] W[k,0,r] = 1
            # U[1,j,r] V[j,k,r] W[k,1,r] = 1
            
            # We can set U[1,j,r] = 1, W[k,1,r] = 1.
            # Then U[0,j,r] V[j,k,r] W[k,0,r] = 1 and U[1,j,r] V[j,k,r] W[k,1,r] = 1.
            # This works with 1 multiplication per (j,k) pair.
            
            # So for each j, k, we use 1 rank.
            # Total rank m * p = 3 * 7 = 21.
            
            U[1*m + j][col] = 1
            # V and W already set
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
