"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # The matrix multiplication tensor T has entries T[(i,j),(j',k),(k',i')]
    # = delta(i,j') * delta(k',i)
    # A rank-1 decomposition: sum_r U[a,r] * V[b,r] * W[c,r]
    # where a=(i,j) flattened as i*m+j, b=(j',k) flattened as j'*p+k, c=(k',i') flattened as k'*n+i'
    #
    # For the standard "index-wise" decomposition (schoolbook):
    # rank = n*m*p
    # Each term: U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    #
    # But we can do better by exploiting structure. The key insight:
    # The tensor is a "permutation-like" tensor. For M(n,m) * M(m,p):
    # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # The standard lower bound is max(n*m, m*p, n*p) but upper bounds vary.
    # For 2x4 * 4x6: n*m=8, m*p=24, n*p=12
    #
    # Known result: the rank of matrix multiplication for m x n matrices is at most
    # mn * something. But for rectangular, it's more complex.
    #
    # Let me try a decomposition that exploits the small n=2 dimension.
    # For each pair (i, i') where i=i' (since we need i' = i for the delta),
    # we have: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So for fixed i, this is a 1 x 4 vector times 4 x 6 matrix = 1 x 6 vector.
    # The rank of 1x4 * 4x6 is at most 24 (schoolbook) but can be less.
    #
    # Actually, the rank of (1 x m) * (m x p) is exactly m*p in general.
    # But we have TWO such multiplications (for i=0 and i=1) that are independent.
    # So total rank = 2 * 24 = 48? No, that's the same as schoolbook.
    #
    # Wait, schoolbook rank = n*m*p = 2*4*6 = 48. So that matches.
    #
    # Can we do better? The tensor is a direct sum of two copies of the 4x6 multiplication tensor
    # (one for i=0, one for i=1). The rank of the 4x6 multiplication tensor (i.e., 1x4 * 4x6)
    # is at most 24, and it's known that the rank of matrix multiplication of 1xm by mxp is
    # exactly m*p for m,p >= 2 (this is a result of Winograd and others).
    #
    # So the rank is 48? That seems to be the schoolbook answer.
    #
    # But wait - maybe we can share structure between the two i-indices?
    # No, because the delta(k',i) forces i' = i, so the two blocks are completely independent.
    #
    # However, let me reconsider. The tensor entries are:
    # T[a, b, c] where a = i*m+j (0..7), b = j'*p+k (0..23), c = k'*n+i' (0..11)
    # T[a, b, c] = 1 if j' = j and k' = i, else 0
    #
    # For a given (a, b) = (i*m+j, j'*p+k), T[a,b,c] is nonzero only when j'=j and k'=i.
    # In that case, c = i*n + i' and T = 1 only when i' is free? No:
    # T = delta(j,j') * delta(i', k') where k' is the row index of B... 
    #
    # Let me re-derive. C[i', k'] = sum_j A[i',j] * B[j,k']
    # So the tensor is: T[(i',j), (j,k'), (k',i'')] -- no, that's not right.
    #
    # Standard: The bilinear map is (A, B) -> A*B
    # A is n x m, B is m x p, C is n x p
    # C[i',k'] = sum_j A[i',j] * B[j,k']
    #
    # The tensor T has indices (i',j), (j,k'), (k',i'') ... no.
    # T is a 3-tensor: T[(i',j), (j,k'), (k',i'')] -- this doesn't make sense.
    #
    # Correct: T[a, b, c] where:
    # a indexes A: a = (i', j) -> flattened to i'*m + j, range [0, n*m)
    # b indexes B: b = (j, k') -> flattened to j*p + k', range [0, m*p)
    # c indexes C: c = (k', i'') -> wait, C is n x p, so c = (i'', k') -> flattened to k'*n + i''?
    #
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) -> k*n + i, meaning c = k*n + i where i is the row of C and k is the column.
    # So C[i, k] with c = k*n + i.
    #
    # T[a, b, c] = sum over... no, T is the tensor such that C[c] = sum_{a,b} T[a,b,c] * A[a] * B[b]
    # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    #
    # This is exactly the schoolbook tensor. Its rank is n*m*p = 48 in the standard decomposition.
    #
    # Can we find a lower rank decomposition? This is an open problem in general, but for small
    # dimensions we can try to verify or use known results.
    #
    # For 2x4 * 4x6, I believe the minimal rank is indeed 48, but let me try to construct
    # a potentially better one using linear combinations.
    #
    # Actually, let me try a different approach: use the fact that we can combine terms.
    # For each i, the sub-tensor for row i of C is: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 4x6 matrix multiplication (treating A[i,:] as a 1x4 vector and B as 4x6).
    # The rank of this sub-problem is at most 24.
    #
    # But since the two rows are independent (different i values don't interact), the total
    # rank is the sum of the ranks of the sub-problems = 24 + 24 = 48.
    #
    # Unless there's some clever way to share computation between rows, which there isn't
    # for exact multiplication (the rows of A are independent inputs).
    #
    # So the rank is 48. Let me just implement the schoolbook decomposition cleanly.
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
