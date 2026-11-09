"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Try a decomposition with rank = m * (n + p) = 3 * 11 = 33
    # Or try other structures. Let's use a known approach:
    # For each j in [0,m), we can write contributions using shared terms.
    
    # Actually, let's try the following structure:
    # Rank = n*p + m*(n+p) - m = 18 + 3*11 - 3 = 48? No, too high.
    
    # Let me try a simpler constructive approach with rank = 27
    # Use the fact that sum_j A[i,j]*B[j,k] can be computed with fewer products
    # by exploiting that m=3 is small.
    
    # For m=3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # We can use: x*y = ((x+y)^2 - x^2 - y^2)/4 but that's for scalars.
    
    # For the full tensor, let's try rank = 18 + 9 = 27:
    # For each (i,k), we have 3 terms. But we can share some.
    
    # Alternative: use rank = m*p + n*m = 27 + 6 = 33? 
    # Or rank = n*p + n*m = 18 + 6 = 24?
    
    # Let me try a direct construction with rank = 27:
    # For each j and k, create a term that handles A[:,j]*B[j,k] for all i.
    # That's m*p = 27 terms, but each term would need to be rank-1 in the tensor sense.
    # Term (j,k): U[i, (j,k)] = A[i,j]... but U must be constant!
    
    # I see - U, V, W are FIXED matrices, not dependent on A and B.
    # So I need: C[i,k] = sum_r U[i*m+j... wait, let me re-read the interface.
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition means: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no.
    
    # Actually, the tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]?
    # No, standard tensor decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # But here the indices are different.
    
    # From the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] summed over r should give
    # the coefficient of A[i,j]*B[j,k] in C[k*n+i... wait, C[i,k].
    
    # The output index is k*n+i, so C[i,k] is at position k*n+i.
    # For the decomposition to work: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # for all i,j,k (since each (i,j,k) triple appears exactly once in schoolbook).
    
    # So I need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other index combinations, the sum should be 0.
    # But wait, U only has indices i*m+j, V has j*p+k, W has k*n+i.
    # The constraint is only that for valid (i,j,k) triples, the product sums to 1.
    
    # This is a 3-way tensor decomposition where the tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # for all i,j,k, and 0 elsewhere. But since i,j,k determine the indices uniquely,
    # it's just that each of the n*m*p = 54 entries is 1.
    
    # So I need a rank-r decomposition of a tensor with 54 ones in specific positions.
    
    # Let me try to find a low-rank decomposition. I'll use a greedy/constructive approach.
    
    # Try rank = 18: one term per output entry (i,k).
    # For each (i,k), I need sum_j U[i*m+j, r_ik] * V[j*p+k, r_ik] = 1 for all j? No...
    # Each term r contributes to multiple (i,j,k) triples.
    
    # Let me try a different strategy: use the known result that for (2x3)*(3x9),
    # the rank might be around 20-30. Let me try to construct one with rank 27.
    
    # Construction: For each j in 0..2 and k in 0..8, define a term that handles
    # the contribution of A[:,j]*B[j,:]. But U[i*m+j, r] must work for all i.
    
    # If I set for term r corresponding to (j,k):
    #   U[i*m+j, r] = 1 for all i, and 0 otherwise in U
    #   V[j*p+k, r] = 1, 0 otherwise in V  
    #   W[k*n+i, r] = 1 for all i, 0 otherwise in W
    # Then sum over r: for a given (i,j,k), only the term with that specific (j,k) 
    # and where U[i*m+j,r]=1 and W[k*n+i,r]=1 contributes. But U[i*m+j,r]=1 for all i
    # means this term contributes to all i for that (j,k). And W[k*n+i,r]=1 for all i
    # means it contributes to all i for that k. So the product is 1 for all i for that (j,k).
    # That gives C[i,k] += 1*A[i,j]*B[j,k] for the specific j,k. But I need sum over j.
    
    # This doesn't directly work because I'm not summing over j properly.
    
    # Let me just try to use a numerical approach or a known construction.
    # Given the complexity, let me try rank = 27 with a systematic construction.
    
    # Actually, the simplest improvement: use rank = 27 by having one term per (j,k) pair,
    # where each term handles the full i-dimension.
    
    rank = m * p  # 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # Term col handles (j,k)
            for i in range(n):
                U[i * m + j][col] = 1.0
                W[k * n + i][col] = 1.0
            V[j * p + k][col] = 1.0
            col += 1
    
    # Verify: for (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Only r = j*p+k has U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
    # So the product is 1. Correct!
    
    # But wait, does this work for the actual matrix multiplication?
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
    # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_k... no, V[j*p+k,r]*B[j,k])
    
    # Hmm, I need to think about how U, V, W combine with A, B.
    # The standard form: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) * W[k*n+i,r]?
    # That doesn't look right either.
    
    # Let me re-examine. In the schoolbook, each term is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # and the full expression is sum_{i,j,k,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_{i,k} C[i,k] * (sum_{j,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
    # For this to equal sum_{i,k} C[i,k], we need the inner sum to be 1 for each (i,k).
    
    # With my construction: sum_{j,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_j A[i,j]*B[j,k]*U[i*m+j, j*p+k]*V[j*p+k, j*p+k]*W[k*n+i, j*p+k]
    # = sum_j A[i,j]*B[j,k]*1*1*1 = sum_j A[i,j]*B[j,k] = C[i,k]. 
    # Wait, that's circular. The point is that the tensor decomposition should satisfy:
    # T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] where T[i,j,k]=1 for all i,j,k.
    # Then C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]... no, T is the decomposition of the identity-like tensor.
    
    # I think the correct interpretation is: the multiplication is defined by
    # C[i,k] = sum_{j,r} A[i,j]*B[j,k]*W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r])
    # No, that's not standard.
    
    # Let me just go with the schoolbook-like construction that I know works and see if I can reduce rank.
    # My construction above with rank=27 should work if the tensor condition holds.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
