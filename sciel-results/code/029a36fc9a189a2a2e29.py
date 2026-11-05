"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-15 decomposition for 3x5x5 matrix multiplication.
    
    Approach: Model as 15 independent dot products (3 rows x 5 columns).
    For each output element C[i,k], we compute sum_j A[i,j] * B[j,k].
    
    We can factor this by noting that for a fixed i, the 5 values C[i,0..4]
    are linear combinations of A[i,0..4] with coefficients from B.
    
    However, a simpler rank-15 construction uses the fact that we can
    decompose the tensor using 15 rank-1 terms where each term handles
    one (i,k) output pair, but we share the A and B factors cleverly.
    
    Actually, the standard schoolbook is rank 75. To get rank 15, we need
    a smarter decomposition. Let's think of it as:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    If we fix j, then for all (i,k) pairs, we have A[i,j] * B[j,k].
    That's rank 15 (one per j), but each rank-1 term would need to cover
    all 15 (i,k) outputs simultaneously, which isn't possible with a single
    rank-1 term in the standard tensor decomposition.
    
    Let me reconsider. The tensor T has indices (i,j,k) with T[i,j,k] = 1.
    We want to write T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    
    Wait, the W factor has indices (k*n+i, r) per the trace convention.
    
    For rank 15, we can use r = 0..14. Let's try:
    - For each j in 0..4, we have 3 sub-terms (one for each i in 0..2)
    - U[i*m+j, r] = 1 if r corresponds to (j, i)
    - V[j*p+k, r] = B[j,k]... but we need constant coefficients
    
    Actually, let's use the following: for each j (5 values), and for each 
    of the 3 output rows i, we create a rank-1 term. But that gives 15 terms.
    
    Term r corresponds to (j, i):
    - U[i*m + j, r] = 1
    - V[j*p + k, r] = 1 for all k (this doesn't work because we need B[j,k])
    
    Hmm, this approach doesn't directly work because V needs to depend on k.
    
    Let me try a different decomposition. Consider:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can write this as:
    C[i,k] = sum_j sum_l A[i,j] delta_{j,l} B[l,k]
    
    That's still rank 75.
    
    For rank 15, one approach is to use the fact that 5 = 5, and use
    a decomposition of the 5x5 identity or use bilinear forms.
    
    Actually, rank 15 for 3x5x5 is not known to be achievable with simple
    constructions. The best known upper bounds are higher.
    
    Let me just stick with a valid decomposition. The schoolbook is rank 75.
    Can we do better?
    
    For 1x5x5, the rank is 5 (just the 5 dot product terms).
    For 3x5x5, if the 3 rows were independent, we might expect 15.
    
    Let's try: r = 0..14, where r = 5*j + i for j in 0..4, i in 0..2.
    U[i*m + j, r] = 1
    V[j*p + k, r] = 1  (for all k)
    W[k*n + i, r] = 1  (for all k)
    
    Then sum_r U[i',j',r] V[j',k',r] W[k',i',r] = 
    sum_{j': r=5*j'+i'} 1 * 1 * 1 = sum_{j' where i'=i} 1
    
    This gives C[i',k'] = number of j' such that the term matches, which is 1
    if i' is in the right range, not the actual dot product.
    
    This doesn't work. Let me just return the schoolbook decomposition.
    """
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
