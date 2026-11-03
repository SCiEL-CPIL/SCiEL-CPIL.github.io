"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Use the "ladder" or "shifted" decomposition technique for n=2.
    # The idea is to construct a rank-2mp decomposition by summing two rank-mp
    # decompositions, but exploit linear dependencies to potentially reduce rank.
    
    # For n=2, we can use the following approach:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # We'll try to construct a decomposition with rank less than 2*m*p = 60.
    # One known technique is to use a "shifted" basis where we can share
    # some rank-1 terms between the two output rows.
    
    # Let's try a constructive approach using a ladder structure.
    # We'll use rank = 2*m*p - m = 55 as a starting point and verify correctness.
    
    # Actually, let me try a different approach: use a rank-1 tensor construction
    # that exploits the small n dimension.
    
    # For each j and k, we need to contribute to both C[0,k] and C[1,k].
    # We can use a "ladder" where we have m*p base terms and then add
    # additional terms to handle the coupling between the two rows.
    
    # Let's try rank = 2*m*p - m = 55
    rank = 2 * m * p - m  # 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the ladder decomposition
    col = 0
    
    # First, add the "base" terms for i=0 (first row)
    for j in range(m):
        for k in range(p):
            if col >= rank:
                break
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Then add terms for i=1 (second row), but we can potentially share
    # some structure. For now, add them directly.
    for j in range(m):
        for k in range(p):
            if col >= rank:
                break
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    # We've used 2*m*p = 60 terms, but rank is 55. This won't work directly.
    # Let me reconsider the approach.
    
    # Instead, let's use a proper ladder decomposition.
    # Reset and try again with a different strategy.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a shifted decomposition:
    # For each "ladder level" t from 0 to m-1, we process a diagonal of the j-k plane.
    
    col = 0
    for t in range(m):
        # For each k, we can have a term that contributes to both rows
        for k in range(p):
            if col >= rank:
                break
            j = t  # This is a simplified ladder; may need adjustment
            
            # This approach is getting complex. Let me fall back to a 
            # verified lower rank if possible, or use schoolbook.
            pass
    
    # Given the complexity, let me try a different known approach:
    # Use the fact that for 2xmxp, we can sometimes achieve rank 2mp - 1
    # or similar by careful construction.
    
    # For now, return schoolbook to ensure correctness, but with a note
    # that a better decomposition may exist.
    
    rank = n * m * p
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
