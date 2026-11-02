"""Rewritten program for M_{2,2,15} using block-wise decomposition."""

def construct():
    # Dimensions: A is 2x2, B is 2x15, C is 2x15
    n, m, p = 2, 2, 15
    
    # The tensor M_{2,2,15} decomposes into 15 independent M_{2,2,1} blocks.
    # Each block handles C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} for a fixed k.
    # The rank of M_{2,2,1} is 4 (optimal).
    # Total rank = 15 * 4 = 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # Size 4 x 60
    V = [[0] * rank for _ in range(m * p)]  # Size 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # Size 30 x 60
    
    col_idx = 0
    # Iterate over the 15 columns of B (index k)
    for k in range(p):
        # For each k, we perform the standard 4-term multiplication of A (2x2) by B_k (2x1)
        # The 4 terms correspond to (i,j) pairs in A: (0,0), (0,1), (1,0), (1,1)
        # But standard schoolbook for C_{i,k} sums over j.
        # C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
        # C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k}
        
        # Term 1: A_{0,0} * B_{0,k} contributes to C_{0,k}
        # U index for A_{0,0} is 0*2+0 = 0
        # V index for B_{0,k} is 0*15+k = k
        # W index for C_{0,k} is k*2+0 = 2k
        
        U[0][col_idx] = 1
        V[k][col_idx] = 1
        W[2*k][col_idx] = 1
        col_idx += 1
        
        # Term 2: A_{0,1} * B_{1,k} contributes to C_{0,k}
        # U index for A_{0,1} is 0*2+1 = 1
        # V index for B_{1,k} is 1*15+k = 15+k
        # W index for C_{0,k} is 2k
        
        U[1][col_idx] = 1
        V[15+k][col_idx] = 1
        W[2*k][col_idx] = 1
        col_idx += 1
        
        # Term 3: A_{1,0} * B_{0,k} contributes to C_{1,k}
        # U index for A_{1,0} is 1*2+0 = 2
        # V index for B_{0,k} is k
        # W index for C_{1,k} is 2k+1
        
        U[2][col_idx] = 1
        V[k][col_idx] = 1
        W[2*k+1][col_idx] = 1
        col_idx += 1
        
        # Term 4: A_{1,1} * B_{1,k} contributes to C_{1,k}
        # U index for A_{1,1} is 1*2+1 = 3
        # V index for B_{1,k} is 15+k
        # W index for C_{1,k} is 2k+1
        
        U[3][col_idx] = 1
        V[15+k][col_idx] = 1
        W[2*k+1][col_idx] = 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
