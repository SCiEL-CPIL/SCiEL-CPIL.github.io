"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Use a structured decomposition
    # For 3x3x3, best known rank is 23
    # For the remaining 4 columns, use schoolbook (4 terms each for 3x3 part)
    # Total: 23 + 4*3 = 35
    
    # Rank-23 decomposition for 3x3x3 (one known construction)
    # U3: 9x23, V3: 9x23, W3: 9x23
    # This is complex to hardcode, so let's try a different approach
    
    # Alternative: use rank 49 decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Group terms by exploiting linearity
    
    # Try rank 35: 23 for first 3 cols + 12 for last 4 cols
    # But I need actual rank-23 factors
    
    # Let's try a simpler but valid improvement: rank 54
    # Use schoolbook but with some sharing
    
    # Actually, let me try to construct rank 49
    # Idea: C[i,k] = sum_j A[i,j]*B[j,k]
    # For each i, C[i,:] = A[i,:]*B (row i of A times B)
    # This is a 1x3 times 3x7 = 1x7
    
    # Let me try rank 42 = 6*7
    # Or rank 49 = 7*7
    
    # Simple approach: for each of the 7 columns of B, use a rank-7 decomposition
    # for multiplying a 3x3 matrix by a 3x1 vector? No, that's not right.
    
    # Let me just use a direct construction
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_j (A[i,j])*(B[j,k])
    
    # The tensor has 3*3*7 = 63 terms in schoolbook
    # To reduce rank, we need to find linear dependencies
    
    # Try: rank 49
    rank = 49
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(21)]
    W = [[0]*rank for _ in range(21)]
    
    # Simple construction: for each i (0-2) and k (0-6), 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Assign each (i,k) pair a set of 7 terms (one for each j)
    # But that's still 63.
    
    # Let me try a different grouping
    # For fixed i, C[i,:] is a linear combination of rows of B
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # So C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    
    # This means for each i, we need 3 terms (one for each j)
    # Total: 3*3 = 9 terms? No, that's not how it works in tensor decomposition.
    
    # In tensor decomposition, each term is a product of a vector in U, V, W
    # Term t: U[:,t] * V[:,t] * W[:,t]
    # Contribution to C[i,k] from term t: U[i,j,t] * V[j,k,t] * W[k,i,t]... 
    
    # Wait, I need to be careful about the index conventions.
    # U has shape (n*m, rank) = (9, rank), indexed by (i,j)
    # V has shape (m*p, rank) = (21, rank), indexed by (j,k)
    # W has shape (p*n, rank) = (21, rank), indexed by (k,i)
    
    # C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # But this sums over both j and t, which doesn't make sense.
    
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # where T[i,j,k] = delta for the multiplication tensor
    # For matrix multiplication, T[i,j,k] = 1 if... 
    
    # The multiplication tensor satisfies:
    # C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T is such that T[i,j,k] = 1 (the coefficient)
    # In the decomposition: sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = T[i,j,k]
    # But T[i,j,k] is just 1 for all i,j,k? No, that's not right either.
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k} ... no.
    
    # Let me re-read the convention. W[k*n+i, r] means W is indexed by (k,i)
    # where k is from B's columns and i is from A's rows.
    
    # The decomposition should satisfy:
    # For the multiplication tensor, we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for the appropriate (i,j,k)
    
    # Actually, the standard form is that the tensor M[i,j,k] = 1 
    # (for the bilinear form of matrix multiplication) is decomposed as:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # But with the W convention W[k*n+i], it's W[k,i] instead of W[i,k]
    
    # So: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
    # where i in [0,3), j in [0,3), k in [0,7)
    
    # Wait, that can't be right - it would mean the tensor is all ones.
    
    # I think the correct interpretation is:
    # The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # (since each product term A[i,j]*B[j,k] contributes to C[i,k])
    
    # No wait, that's not a tensor in the usual sense. Let me reconsider.
    
    # In matrix multiplication tensor decomposition, the tensor is:
    # T[i,j,k] = delta_{i,k} ... no.
    
    # Actually, for multiplying an m×n matrix A by an n×p matrix B:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] where the "1" appears at positions that make this work.
    # Specifically, T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # But T[i,j,k] = 1 for all i,j,k doesn't seem right.
    
    # Let me look at the schoolbook example:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # for each (i,j,k) triple.
    
    # So for a fixed column c corresponding to (i0,j0,k0):
    # U[i,j,c] = 1 if i=i0 and j=j0, else 0
    # V[j,k,c] = 1 if j=j0 and k=k0, else 0  
    # W[k,i,c] = 1 if k=k0 and i=i0, else 0
    
    # So U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 if (i,j)=(i0,j0) AND (j,k)=(j0,k0) AND (k,i)=(k0,i0)
    # = 1 if i=i0, j=j0, k=k0
    # = 1 only when (i,j,k) = (i0,j0,k0)
    
    # So sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 if there exists c with (i,j,k)=(i0,j0,k0) for that c
    # = 1 for each (i,j,k) in schoolbook
    # = 0 otherwise
    
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k), and 0 otherwise.
    # But since all (i,j,k) with i in [0,3), j in [0,3), k in [0,7) are valid,
    # T is just the all-ones tensor of shape (3,3,7).
    
    # Wait, that means we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    # with i in [0,3), j in [0,3), k in [0,7)
    
    # And for invalid combinations, it should be 0, but since U,V,W are only defined
    # for valid indices, we only care about valid (i,j,k).
    
    # So the problem is: find minimum rank such that there exist U (9×rank), V (21×rank), W (21×rank)
    # with sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    
    # This is a rank-1 approximation problem for the all-ones tensor!
    
    # The all-ones tensor of shape (3,3,7) has rank 1! Because:
    # T[i,j,k] = 1 = 1*1*1
    
    # So U[i,j,r] = 1, V[j,k,r] = 1, W[k,i,r] = 1 for r=0, and 0 for r>0
    # gives rank 1!
    
    # But wait, that can't be right. Matrix multiplication tensor is not all ones.
    
    # I'm confused. Let me re-read the problem.
    
    # "Matrix multiplication 3,3,7" - multiply 3x3 by 3x7
    # The output is 3x7.
    
    # In the schoolbook: for each (i,j,k), we have a term that contributes A[i,j]*B[j,k] to C[i,k]
    # The tensor should encode which (i,j,k) triples contribute.
    
    # I think the tensor is NOT all ones. The tensor T[i,j,k] should be such that
    # the bilinear form gives the correct matrix multiplication.
    
    # Actually, I think I was overcomplicating this. The tensor for matrix multiplication
    # A (m×n) * B (n×p) = C (m×p) is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] should contribute to C[i,k]
    # which is always true, so T[i,j,k] = 1 for all valid i,j,k.
    
    # But that gives rank 1, which is clearly wrong.
    
    # I must be misunderstanding the tensor convention.
    
    # Let me look at it differently. In the standard tensor decomposition for matrix
    # multiplication, the tensor is defined as:
    # T[i,j,k] = delta_{i,k}... no.
    
    # Actually, I think the issue is that the tensor represents the structure of the
    # computation, not the values. Let me reconsider.
    
    # For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Each term in the sum is a product of one entry of A and one entry of B.
    # The tensor T has T[i,j,k] = 1 indicating that A[i,j] and B[j,k] multiply
    # to contribute to C[i,k].
    
    # In the tensor decomposition:
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # For schoolbook, each (i,j,k) gets its own term, so T[i,j,k] = 1.
    
    # But if T is all ones, then rank 1 works: U=V=W=1.
    
    # This can't be right. There must be something I'm missing.
    
    # OH WAIT. I think the tensor is not T[i,j,k]=1. The tensor encodes the
    # multiplication structure differently.
    
    # Let me reconsider. In the standard definition, the matrix multiplication
    # tensor for m×n * n×p is an order-3 tensor of shape (m*n, n*p, p*m) or similar,
    # not (m,n,p).
    
    # I think the indices are:
    # - First mode: (i,j) for A, size m*n
    # - Second mode: (j,k) for B, size n*p  
    # - Third mode: (i,k) for C, size m*p
    
    # And the tensor T[(i,j),(j,k),(i,k)] = 1 if the indices are consistent,
    # i.e., the j in A[i,j] matches the j in B[j,k], and the i in C[i,k] matches
    # the i in A[i,j], and the k in C[i,k] matches the k in B[j,k].
    
    # Since all indices are consistent by definition (j is shared, i and k are shared),
    # T is again all ones for valid index combinations.
    
    # I'm clearly misunderstanding something fundamental.
    
    # Let me just go with the schoolbook approach and try to optimize it.
    # The schoolbook gives rank 63. Let me try to find a better construction.
    
    # Actually, you know what, let me just implement a reasonable heuristic
    # and see if it works. I'll try rank 49 with a structured approach.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness,
    # and add a comment about potential improvements.
    
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
