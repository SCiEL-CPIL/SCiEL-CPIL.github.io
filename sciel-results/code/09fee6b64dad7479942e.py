"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-32 decomposition using a structured block approach.
    # For 2x4 * 4x8, we can use a rank of 32 by leveraging the rectangular structure.
    #
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # has entries T[i,j,k] = delta[i,j,k] (i.e., 1 if the (i,k) entry of C gets contribution
    # from A[i,j] * B[j,k]).
    #
    # A rank-r decomposition means:
    #   T[i,j,k] = sum_{t=0}^{r-1} U[i,j,t] * V[j,k,t] * W[k,i,t]
    #
    # For n=2, m=4, p=8, the trivial rank is 64.
    # We can do better by grouping terms.
    #
    # Strategy: Use the fact that p=8=2*4, m=4=2*2, n=2.
    # Group the j dimension into 2 groups of 2, and the k dimension into 2 groups of 4.
    #
    # Actually, let's use a simpler structured approach:
    # For each pair (i, k), we need sum_j A[i,j]*B[j,k] = C[i,k].
    # The trivial decomposition uses one term per (i,j,k) triple, giving rank 64.
    #
    # We can achieve rank 32 by the following construction:
    # For each (i, j_pair, k_pair) where j_pair groups two j's and k_pair groups four k's,
    # we create a term that handles the sub-product.
    #
    # Let's use the approach from Landsberg: for m x n matrices, rank is at most m*n*ceil(log2(max(m,n)))
    # but that's for square. For rectangular 2x4 by 4x8:
    #
    # A concrete rank-32 construction:
    # Split j into 2 blocks of size 2: J0={j0,j1}, J1={j2,j3}
    # Split k into 2 blocks of size 4: K0={k0..k3}, K1={k4..k7}
    # For each i in {0,1}, each j_block in {0,1}, each k_block in {0,1}:
    #   We have a 1x2 * 2x4 subproblem.
    #   The 1x2 * 2x4 product can be done in rank 4 (trivially 1*2*4=8, but we can do 4).
    #   Actually 1x2 * 2x4: C[i,k] = A[i,2b]*B[2b,k] + A[i,2b+1]*B[2b+1,k]
    #   This is a sum of 2 rank-4 terms? No.
    #   The tensor for 1x2 * 2x4 has 1*2*4 = 8 nonzero entries.
    #   We can decompose it in rank 4: for each of the 2 j's, use rank 2 for the 1x1 * 1x4 part?
    #   No, 1x1 * 1x4 is rank 4.
    #   So 1x2 * 2x4 = 2 * (1x1 * 1x4) = rank 8.
    #   That gives total rank = 2 * 2 * 8 = 32.
    #
    # So: for i in range(2), jb in range(2), kb in range(2):
    #   For jj in range(2):  # local j index within block
    #     For kk in range(4):  # local k index within block
    #       j = jb*2 + jj
    #       k = kb*4 + kk
    #       term index t = i*8 + jb*4 + jj*2 + kk  -> this gives 2*2*2*4 = 32 terms
    #
    # Wait, that's just the trivial decomposition reorganized! Rank is still 64? No:
    # 2 * 2 * 2 * 4 = 32 terms. But each term is a rank-1 tensor.
    # Let me recount: i has 2 values, jb has 2, jj has 2, kk has 4.
    # Total = 2*2*2*4 = 32. Yes!
    #
    # But wait, in the trivial decomposition, we have i (2), j (4), k (8) = 64 terms.
    # Here we have i (2), jb (2), jj (2), kk (4) = 32 terms.
    # The mapping j = jb*2 + jj is bijective on j in {0,1,2,3}.
    # The mapping k = kb*4 + kk is bijective on k in {0,...,7}.
    # So this is just a reindexing of the same 64 terms... 
    #
    # No wait: 2*2*2*4 = 32, but i*j*k = 2*4*8 = 64.
    # The issue is that I'm missing a factor. Let me recount:
    # i: 2 values
    # jb: 2 values
    # jj: 2 values  
    # kk: 4 values
    # Product: 2*2*2*4 = 32. But we need to cover all (i,j,k) with i in [0,1], j in [0,3], k in [0,7].
    # j = jb*2+jj covers all 4 values of j. k = kb*4+kk covers all 8 values of k.
    # So (i, jb, jj, kb, kk) -> (i, j, k) is a bijection from {0,1}^2 * {0,1} * {0,1,2,3} to {0,1} * {0,1,2,3} * {0,...,7}.
    # Domain size: 2*2*2*2*4 = 64. Range size: 2*4*8 = 64.
    # I forgot kb! So it's 64 terms again.
    #
    # OK so that doesn't help. Let me think differently.
    #
    # For 2x4 * 4x8, we can use the following:
    # Write A as a 2x4 matrix, B as a 4x8 matrix.
    # C = A*B is 2x8.
    #
    # Use the identity: for any scalar a,b,c,d, we can compute certain products with fewer terms.
    # 
    # Actually, let's use a known result: the rank of the 2x4 by 4x8 matrix multiplication tensor
    # is at most 32. Here's a construction:
    #
    # For each k in range(8), the k-th column of C is A * B[:,k], which is a 2x1 vector.
    # Computing A * v for a 4x1 vector v: this is a 2x4 * 4x1 product.
    # The rank of 2x4 * 4x1 matrix multiplication is 4 (since it's essentially 4 dot products of length 4,
    # but structured: C[0] = sum_j A[0,j]*v[j], C[1] = sum_j A[1,j]*v[j]).
    # This can be done in rank 4: for each j, one term. That's the trivial 4 terms.
    # Can we do better? For 2x4 * 4x1, the tensor has 2*4*1 = 8 entries.
    # The rank is at least 4 (since restricting A to have only one nonzero row gives a 1x4*4x1 product of rank 4).
    # And rank 4 is achievable trivially.
    # So for each k, we need rank 4. Total rank = 8*4 = 32.
    #
    # So the construction is: for each k in range(8) and j in range(4):
    #   U[i,j,t] = 1 if the term corresponds to (i, j, k) and the i-th row of A
    #   V[j,k,t] = 1 if it corresponds to B[j,k]
    #   W[k,i,t] = 1 if it corresponds to C[i,k]
    #
    # This is just the trivial decomposition! Rank = 2*4*8 = 64.
    #
    # I'm going in circles. Let me just use the schoolbook rank of 64, which is correct.
    # The task asks for rank-minimal, and 64 is the trivial upper bound.
    # If a lower rank exists, we'd need a more clever construction.
    #
    # For now, return the schoolbook decomposition with rank 64.
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
