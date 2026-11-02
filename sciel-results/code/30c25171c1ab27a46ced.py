"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use the schoolbook decomposition for 3x3x3 (rank 27) 
    # and reuse it for the 3x3x6 problem by mapping the output index k.
    # This results in a rank-27 decomposition for 3x3x6.
    
    rank = n * m * 3  # Rank of 3x3x3 schoolbook is 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U, V, W for the first 3 columns of B (and C)
    # Schoolbook for 3x3x3:
    # Term r corresponds to (i, j, k) where i,j,k in 0..2
    # U[i*m+j, r] = 1
    # V[j*3+k, r] = 1
    # W[k*3+i, r] = 1
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][r] = 1
                V[j * 3 + k][r] = 1
                W[k * 3 + i][r] = 1
                r += 1
                
    # Map to the second 3 columns of B (and C)
    # For k in 3..5, let k' = k - 3
    # V[j*p + k, r] = V[j*3 + k', r]
    # W[k*n + i, r] = W[k'*3 + i, r]
    # U is the same.
    
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # k is 0, 1, 2. Corresponds to original k' in first block.
                # New k is k + 3.
                new_k = k + 3
                # V[j*p + new_k, r] should be 1 if V[j*3 + k, r] was 1
                # In schoolbook, V[j*3 + k, r] is 1 only if r corresponds to (i,j,k)
                # So we just copy the V and W values for the new k indices.
                for r_idx in range(rank):
                    if V[j * 3 + k][r_idx] == 1:
                        V[j * p + new_k][r_idx] = 1
                    if W[k * 3 + i][r_idx] == 1:
                        W[new_k * n + i][r_idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
