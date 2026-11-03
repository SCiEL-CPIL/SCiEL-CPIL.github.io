"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank-9 decomposition for 1x4 x 4x9 sub-problem, repeated for each of 4 rows
    # Total rank = 4 * 9 = 36
    rank = n * p  # 4 * 9 = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i (0..n-1) and each column k (0..p-1), we have a rank-1 term
    # The term contributes to output W[i,k] via U[i,j] * V[j,k]
    # We need: for fixed i, sum over j of A[i,j] * B[j,k] = C[i,k]
    # Rank-1 decomposition for 1x4 x 4x9: use 9 terms, one per column k
    # Term t (for column k): U[i, j] = 1 if j == k else 0? No...
    # Actually for 1x4 * 4x9, we can do rank 4 (one per row j of B) or rank 9 (one per col k)
    # Rank 9: for each k, term has U[i,j] = A[i,j]... but that's not rank-1 unless we fix j.
    # Let me think differently.
    # For 1x4 * 4x9, the standard decomposition is rank 4: sum over j=0..3 of (A[:,j]) * (B[j,:])
    # But we want rank 9: sum over k=0..8 of something.
    # Actually, 1x4 * 4x9 can be decomposed as: for each k, the k-th column of result is A * B[:,k]
    # That's 4x1 * 1x1 per k, so 4 terms per k, total 4*9=36. But that's just the full rank.
    # 
    # Wait, the directive says "rank-9 decomposition (via 4 independent scalar-vector multiplications)"
    # This suggests: for each row i, decompose the 1x4 * 4x9 product using 9 terms.
    # How? One way: for each k, use a term that isolates column k.
    # Term for (i, k): U[i, j] = delta(j, k) for all j? No, that gives U[i,j] = 1 if j==k.
    # Then V[j, k'] = delta(j, k') * ... hmm.
    # 
    # Let's use: for each i in 0..n-1 and k in 0..p-1, define term t = i*p + k
    # U[i*m + j][t] = 1 if j == k else 0  -- but j goes 0..m-1=0..3, k goes 0..8
    # This doesn't work since k can be > m-1.
    # 
    # Alternative: The rank-9 for 1x4*4x9 means we use 9 rank-1 tensors.
    # For the single-row case (n=1), we need: sum_{t=0}^{8} U[1,4,t] * V[4,9,t] * W[9,1,t]
    # W has shape 9x1, so W[k, 0, t] for k=0..8
    # We want: C[k] = sum_j A[j] * B[j,k]
    # Set: for each t in 0..8, let t correspond to column k=t
    # U[j, t] = A[j] -- but A is variable, so this won't be a fixed decomposition.
    # 
    # I think the correct rank-9 decomposition for 1x4 * 4x9 is:
    # For each k (0..8), term k has:
    #   U[j, k] = 1 for all j (so U is all-ones column)
    #   V[j, k'] = B[j, k'] ... no, V must be fixed.
    # 
    # Actually, for a fixed multiplication A*B, the tensor is fixed. The decomposition is for the
    # bilinear map. Let me reconsider.
    # 
    # The standard way: rank-4 for 1x4*4x9: sum_{j=0}^{3} (e_j^T) * (B[j,:]) where e_j is unit vector
    # That gives rank 4.
    # 
    # For rank 9: we can write it as sum_{k=0}^{8} (A * e_k) * (e_k^T * B) but that's not rank-1.
    # 
    # I think the intended approach is different. Let me just use the trivial rank-36 decomposition
    # which is the same as the current one but with a different indexing, achieving rank 36.
    # 
    # Actually, the current program uses rank = n*m*p = 4*4*9 = 144. We can do much better.
    # The standard decomposition for n x m * m x p has rank n*m*p (trivial) but can be improved.
    # For 4x4 * 4x9, a simple decomposition uses rank 4*9 = 36:
    # For each i (row of A) and k (col of B), term (i,k):
    #   U[i*m + j, (i,k)] = 1 if we're summing over j... 
    # 
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Term (i,k,j) would be rank 1, but we sum over j.
    # To get rank 36: for each (i,k), we need one term. But that term must capture sum_j A[i,j]B[j,k].
    # That's not possible with one rank-1 term unless m=1.
    # 
    # So rank 36 is NOT achievable with this simple approach. The minimal rank for 4x4*4x9 is at least...
    # Let me just keep the trivial decomposition but optimize the indexing.
    # The trivial rank is n*m*p = 144. We can do better with Strassen-like but that's complex.
    # 
    # For now, let's use the straightforward rank = m * (n * p) = 4 * 36 = 144, same as before.
    # Or rank = n * m * p = 144.
    
    # Let's use the standard decomposition with rank = n * m * p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        k = idx % p
        j = (idx // p) % m
        i = idx // (p * m)
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
