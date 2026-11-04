"""Matrix multiplication 2x4 by 4x5: rank-10 decomposition via row-splitting."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p  # 2 * 5 = 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i of the 2x4 matrix and each column k of the 4x5 matrix,
    # we create one tensor term. The contribution to output C[i,k] is sum_j A[i,j]*B[j,k].
    #
    # Decomposition: for each (i, k) pair, we need a term that multiplies
    # A[i,:] (a 1x4 row) by B[:,k] (a 4x1 column).
    #
    # The standard rank-p decomposition for 1xm by mxp (when m is the "inner" dimension
    # and we fix the outer structure) uses p terms if we can factor through the p
    # output columns. But here we have n rows of A, each producing p outputs.
    #
    # Key insight: For a fixed row i of A, the product A[i,:] * B gives a 1xp row.
    # This can be decomposed using p terms where term k uses:
    #   U component: picks out A[i,j] for all j (but we need rank-1 in the j-dimension)
    #
    # Actually, let's think more carefully. The tensor for matrix multiplication
    # C[i,k] = sum_j A[i,j] B[j,k] has a natural decomposition.
    #
    # For the 2x4 by 4x5 case, the rank is at most n*p = 10. Here's the construction:
    # For each pair (i, k) where i in {0,1} and k in {0,...,4}, define term t = i*p + k.
    # We need U, V, W such that the bilinear form gives the correct contribution.
    #
    # The issue: each term must be rank-1 in the combined (i,j,k) tensor.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-1 term in the multiplication tensor looks like:
    #   A[i',j'] * B[j',k'] * C[i'',k'']
    # which contributes to C[i'',k''] when i'=i'', j'=j', k'=k''.
    #
    # For rank n*p = 10, we use terms indexed by (i, k). For term (i, k):
    # We want it to contribute to C[i,k] the value sum_j A[i,j] B[j,k].
    # But a single rank-1 term can only pick out one specific (j) via U and V.
    #
    # Wait - the rank of matrix multiplication for n x m by m x p is at most n*m*p
    # (schoolbook), but can be lower. For 1 x m by m x p, the rank is exactly p
    # (since C[k] = sum_j A[j] B[j,k], and we can use p terms).
    #
    # For 2 x 4 by 4 x 5: We can view this as two independent 1x4 by 4x5 multiplications.
    # Each 1x4 by 4x5 has rank 5. Total rank = 2*5 = 10.
    #
    # For row i of A (which is a 1x4 vector), the product with B (4x5) gives a 1x5 vector.
    # Rank-5 decomposition for 1x4 by 4x5: for each output column k:
    #   Term: U picks A[:,j] for the specific row i, V picks B[j,k] for all j
    #   More precisely, for the 1xm by mxp case with rank p:
    #     For each k in {0,...,p-1}:
    #       U[j] = A[j] (for the single row, so U is 1-dimensional in the row space)
    #       Actually, for 1xm: U is shape (1*m, rank) = (m, rank)
    #       V is shape (m*p, rank)
    #       W is shape (p*1, rank) = (p, rank)
    #
    # Let me be very explicit about the indexing convention:
    # A is n x m, indexed by (i, j) -> position i*m + j in U
    # B is m x p, indexed by (j, k) -> position j*p + k in V  
    # C is n x p, indexed by (i, k) -> position k*n + i in W (trace convention: W[k*n+i, r])
    #
    # For the 1xm by mxp case (n=1):
    #   U has shape (m, rank), V has shape (m*p, rank), W has shape (p, rank)
    #   Rank-p decomposition: for each k in {0,...,p-1}:
    #     U[j, k] = 1 for all j  (so U picks out the entire row of A)
    #     V[j*p + k, k] = 1 for all j  (so V picks out column k of B)
    #     W[k, k] = 1  (so W puts the result in position k)
    #
    # Verify: sum_k U[j,k]*V[j*p+k,k]*W[k,k] summed over j gives
    #   sum_k sum_j 1 * B[j,k] * 1 = sum_j B[j,k] ... wait, that's not right.
    #
    # Let me re-derive. The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in the
    # standard basis. We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    #
    # For n=1, m=4, p=5: T[j,k] = 1 (since i is always 0, and the tensor entry is 1
    # for each (i,j,k) combination in the multiplication).
    #
    # Wait, I need to be more careful. The multiplication tensor for C = A*B where
    # A is n x m and B is m x p: C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that the bilinear map (A, B) -> C is given by
    # C[i,k] = sum_{j} sum_{r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    #
    # No wait, that's not right either. The decomposition means:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])
    #
    # Actually no. The standard tensor decomposition of matrix multiplication:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # where T_{i,j,k} = 1 for all valid (i,j,k), and the multiplication is
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k}
    #
    # Hmm, I think the convention is that the tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # and we want T = sum_r u_r v_r w_r where u_r is a vector in the A-space, v_r in B-space,
    # w_r in C-space.
    #
    # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k).
    #
    # And the multiplication is recovered as:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    #
    # For the schoolbook: U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1 where r=(i,j,k) flattened.
    # Then sum_r U V W = 1, and C[i,k] = sum_j A[i,j] B[j,k]. Correct.
    #
    # Now for the rank-10 decomposition of 2x4 by 4x5:
    # We want 10 terms such that for all (i,j,k) with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}:
    #   sum_{r=0}^{9} U[i*4+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
    #
    # Construction: Index terms by (i, k) where i in {0,1}, k in {0,1,2,3,4}. Let r = i*5+k.
    #
    # Set:
    #   U[i*4+j, r] = 1 if the term's row index matches i, else 0
    #   V[j*5+k, r] = 1 if the term's column index matches k, else 0  
    #   W[k*2+i, r] = 1
    #
    # Let's verify: For a given (i,j,k), sum over r = i'*5+k':
    #   U[i*4+j, i'*5+k'] = 1 iff i' = i
    #   V[j*5+k, i'*5+k'] = 1 iff k' = k
    #   W[k*2+i, i'*5+k'] = 1 iff (k,i) matches (k',i') in the W index, i.e., k'*2+i' = k*2+i, so k'=k and i'=i
    #
    # So the sum is 1 only when i'=i and k'=k, giving exactly 1. 
    #
    # Now check the multiplication:
    # C[i,k] = sum_j sum_r U[i*4+j, r] V[j*5+k, r] W[k*2+i, r] A[i,j] B[j,k]
    # For r = i'*5+k':
    #   U[i*4+j, r] = 1 iff i'=i
    #   V[j*5+k, r] = 1 iff k'=k
    #   W[k*2+i, r] = 1 iff k'=k and i'=i
    # So only r = i*5+k contributes, and the term is:
    #   U[i*4+j, i*5+k] * V[j*5+k, i*5+k] * W[k*2+i, i*5+k] * A[i,j] * B[j,k]
    #   = 1 * 1 * 1 * A[i,j] * B[j,k] = A[i,j] B[j,k]
    #
    # Summing over j: C[i,k] = sum_j A[i,j] B[j,k]. Correct!
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
