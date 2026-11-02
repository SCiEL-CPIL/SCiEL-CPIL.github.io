def _build_rank32():
    """
    Construct Rank 32 decomposition.
    Splits 2x3 * 3x8 into 4 blocks of 2x3 * 3x2.
    Each 2x3 * 3x2 is split into 2x2*2x2 (Rank 7) + 2x1*1x2 (Rank 1).
    Total 8 per block. 4 * 8 = 32.
    """
    n, m, p = 2, 3, 8
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    col = 0
    for b in range(4):
        k0 = 2*b
        k1 = 2*b + 1
        
        # Part 1: A[:, 0:2] * B[0:2, k0:k1+1]
        # This is a 2x2 * 2x2 multiplication.
        # U indices: i*3 + j (j in 0,1)
        # V indices: j*8 + k (j in 0,1, k in k0, k1)
        # W indices: k*2 + i (k in k0, k1, i in 0,1)
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = S_U[i*2+j, r]
                    if val != 0:
                        U[i*3 + j, col+r] += val
            
            # V
            for j in range(2):
                for k in range(2):
                    val = S_V[j*2+k, r]
                    if val != 0:
                        V[j*8 + (k0+k), col+r] += val
            
            # W
            for k in range(2):
                for i in range(2):
                    val = S_W[k*2+i, r]
                    if val != 0:
                        W[(k0+k)*2 + i, col+r] += val
        
        col += 7
        
        # Part 2: A[:, 2] * B[2, k0:k1+1]
        # This is a 2x1 * 1x2 multiplication.
        # U indices: i*3 + 2
        # V indices: 2*8 + k
        # W indices: k*2 + i
        
        # Rank 1 term:
        # U[i*3+2, col] = 1
        # V[2*8+k, col] = 1
        # W[k*2+i, col] = 1
        
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] += 1.0
                V[2*8 + (k0+k), col] += 1.0
                W[(k0+k)*2 + i, col] += 1.0
        
        col += 1

    return U, V, W
