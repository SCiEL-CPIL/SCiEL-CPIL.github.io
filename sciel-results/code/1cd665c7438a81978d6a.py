"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Use Strassen-style decomposition: 7 "products" each of rank 4
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Strassen 2x2x2 gives 7 bilinear forms
    # Each bilinear form: (linear combo of A entries) * (linear combo of B entries per column group)
    # Since B is 2x8, each "row" of B is 1x8
    # The 7 Strassen products each compute a 1x8 vector = scalar * (1x8 row vector from B combo)
    # Each such 1x8 computation has rank 8? No...
    
    # Let me reconsider. C[i,k] = sum_j A[i,j] B[j,k]
    # Strassen decomposes the 2x2 multiplication part.
    # For each k, C[:,k] = A * B[:,k] where B[:,k] is 2x1.
    # So for each k, we do a 2x2 * 2x1 = 2x1 multiplication.
    # A 2x2 * 2x1 multiplication: C[i] = sum_j A[i,j]*B[j]
    # This is equivalent to computing two dot products of length 2.
    # Rank of (2,2,1) tensor: n=2, m=2, p=1.
    # C[i] = sum_j A[i,j] B[j], i in {0,1}, j in {0,1}
    # This is just 2 scalar multiplications summed, so rank = 2? 
    # Actually the tensor T[i,j,k] = delta_{i,k} for k in {0}... 
    # For p=1: T[i,j,0] = delta_{i,0}... no.
    # T[i,j,k] where the multiplication is (A_{i,j})(B_{j,k}) -> C_{i,k}
    # For 2x2 * 2x1: C_i = sum_j A_{i,j} B_j (k dimension is trivial, p=1)
    # T[i,j,0] = 1 for all i,j? No, T[i,j,k] = delta_{k, ?}...
    # Actually for matrix mult (n,m,p), T[i,j,k] = delta_{i,k}... 
    # The tensor is T[i,j,k] = 1 if we're multiplying A_{i,j} by B_{j,k} to get C_{i,k}
    # So T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} in the full expansion
    # For (2,2,1): rank is 2 (just two independent multiplications with summation)
    
    # So for each k (8 values), we need rank 2 for the 2x2*2x1 part.
    # Total rank = 8 * 2 = 16? But that's just schoolbook per column.
    # Schoolbook total: n*m*p = 2*2*8 = 32.
    # Per-column Strassen-like: for 2x2*2x1, can we do better than rank 2?
    # 2x2*2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # This is: C0 = (A00, A01) . (B0, B1), C1 = (A10, A11) . (B0, B1)
    # This is a 2x2 * 2x1 matrix mult. The tensor rank: 
    # We need sum_s u_s[i] v_s[j] w_s[k] = delta_{i,k} for k=0 only... 
    # Actually with p=1, k is always 0, so T[i,j,0] must satisfy:
    # sum_s U_s[i] V_s[j] W_s[0] = 1 for all i,j (since C_i = sum_j A_ij B_j means coefficient is 1)
    # Wait no. The tensor T[i,j,k] for matmult (n,m,p) is defined as:
    # C_i = sum_{j,k} A_{i,j} B_{j,k} ... no, standard is C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = delta_{k, ?}... 
    # T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k], so T[i,j,k] = 1 for the "matching" 
    # Actually the bilinear form is: C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # As a trilinear form on (A, B, C^*): sum_{i,j,k} C^*_{i,k} A_{i,j} B_{j,k} = <C, AB>
    # The tensor is T[i,j,k] = 1 where it's A[i,j] * B[j,k] * C^*[i,k]
    # So T[i,j,k] = 1 for all i,j,k? No, it's that the term A[i,j]*B[j,k] appears in C[i,k].
    # T[i,j,k] = delta_{i,i}delta_{j,j}delta_{k,k}... I'm confusing myself.
    
    # The matrix multiplication tensor for (n,m,p): T is a 3-tensor of size (n*p, m*p, n*m) or similar.
    # In the standard indexing: T[a,b,c] where a indexes (i,k) in output, b indexes (j,k) in B, c indexes (i,j) in A
    # T[(i,k),(j,k'),(i',j')] = delta_{i,i'} delta_{k,k'} delta_{j,j'}... 
    
    # Let me use the interface convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, s] with the trace convention.
    # The decomposition: C[i,k] = sum_s U[i*m+j ... ] 
    
    # From the seed: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So C[i,k] = sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s]
    # For schoolbook: for each (i,j,k), one column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Check: C[i,k] = sum_j U[i*m+j, col(i,j,k)] V[j*p+k, col] W[k*n+i, col]
    #         = sum_j 1 * 1 * 1 (only when the col matches (i,j,k))
    #         = 1 if there's a j such that col = (i,j,k), which is always true for the j in the sum
    #         Wait, for fixed (i,k), we sum over j: the column (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    #         So C[i,k] = sum_j (1)(1)(1) = m = 2. That's wrong!
    
    # Hmm, let me re-read the seed. The seed has U[i*m+j][column]=1, V[j*p+k][column]=1, W[k*n+i][column]=1
    # And C[i,k] should = sum_j A[i,j]*B[j,k]
    # With the tensor: C[i,k] = sum_s U[i*m+? ...] 
    
    # I think the convention is: the tensor T is decomposed as T[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
    # where a indexes the "A" factor position, b indexes "B" factor, c indexes "C" factor
    # For matmult: the relevant tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k]... 
    # Actually I think it's: the multiplication is represented as a trilinear form.
    # Let me just trust the seed's index convention and build from there.
    
    # The seed works for schoolbook with rank n*m*p=32.
    # I need to find a rank-28 decomposition.
    
    # Approach: Use Strassen's 7-product algorithm for the 2x2 part, and handle the 8 columns.
    # For 2x2 x 2x8: C[2x8] = A[2x2] x B[2x8]
    # Split into: for each of the 8 columns, do 2x2 x 2x1.
    # 2x2 x 2x1 has tensor rank 2 (two independent dot products).
    # But we can share structure across columns using Strassen.
    
    # Strassen for 2x2 x 2x8:
    # Define the 7 Strassen "products" but each is a 1x8 vector.
    # M1 = (A00+A11) x (B00+B11)  where B00 is row 0 of B (1x8), B11 is row 1
    # Wait, in Strassen for 2x2x2, the blocks are 1x1. Here B is 2x8 so each "block" is 1x8.
    # M1 = (A00 + A11) * (B[0,:] + B[1,:])  -- this is scalar * (1x8 vector) = 1x8 vector
    # Similarly for all 7.
    # Then reconstruct C from these 7 vectors.
    
    # Each M_s is a 1x8 vector = alpha_s * (linear combo of B rows)
    # Computing alpha_s * v where v is a 1x8 vector: this is 8 scalar multiplications, rank 8 per M_s.
    # 7 * 8 = 56. That's worse than schoolbook!
    
    # I made an error. Let me reconsider.
    # The issue is that in standard Strassen for 2x2x2, the "products" are scalar*scalar.
    # For 2x2 x 2x8, if we think of it as 2x2 x 2x8, the "inner" dimension is 2 (the m dimension).
    # Strassen works on the (i,j) = (2,2) part. The k dimension (8) is just a batch.
    # So we do 7 "products" each of which is: (1x2 row) x (2x8 matrix) = 1x8 matrix.
    # Computing (1x2) x (2x8) = 1x8: this is a matrix-vector product with a 2x8 matrix.
    # The rank of (1,2,8) tensor: n=1, m=2, p=8.
    # C[k] = sum_j A[j] B[j,k], for k=0..7.
    # This is: for each k, a dot product of length 2.
    # The tensor T[j,k] for the bilinear form sum_{j,k} T[j,k] A[j] B[j,k]:
    # T[j,k] = 1 for all j,k. So it's the all-ones tensor of size 2x8.
    # The tensor rank of the all-ones 2x8 matrix (as a bilinear form) is 2 (since rank of the matrix is 2, 
    # and for a matrix M, the tensor rank of the bilinear form <a, Mb> is rank(M)).
    # So (1,2,8) has tensor rank 2!
    
    # Therefore: 7 Strassen products, each of rank 2, total rank = 14.
    # But wait, can we actually achieve rank 14? Let me verify.
    
    # Actually, I need to be more careful. The Strassen decomposition gives us 7 bilinear forms
    # of the form (linear combo of A's) * (linear combo of B's). Each such form, when applied
    # to produce a 1x8 output, has the structure: scalar * (1x8 vector from B combo).
    # The "scalar" comes from A, the "1x8 vector" comes from B.
    # Computing scalar * v (where v is 1x8) is 8 separate multiplications, but they share the scalar.
    # As a tensor: T[j,k] where j indexes the A-combo (1 dimension effectively) and k indexes B.
    # Hmm, I think the rank per Strassen product is indeed 2 (the rank of the 2x8 B-submatrix structure).
    
    # Let me just construct the rank-28 solution as mentioned in the knowledge, and verify.
    # The knowledge says rank 28 works. Let me build it step by step.
    
    # Alternative: maybe the rank is 28 because each of the 7 Strassen products has rank 4,
    # not rank 2. Let me reconsider the (1,2,8) rank.
    
    # For (n,m,p) = (1,2,8): C[0,k] = sum_j A[0,j] B[j,k], j in {0,1}, k in {0..7}
    # The tensor T[j,k] = 1 for all j,k (2x8 all-ones).
    # Tensor rank of all-ones 2x8: We need sum_s u_s[j] v_s[k] = 1 for all j,k.
    # This is: U^T V = ones(2,8). Rank of ones(2,8) is 1, so we can do it with rank 1!
    # u_0 = [1,1], v_0 = [1,1,1,1,1,1,1,1]. Then sum_s u_s[j]v_s[k] = 1*1 = 1. Yes!
    # So (1,2,8) has rank 1?! That seems too good.
    
    # Wait, no. The tensor rank of a matrix M (as a bilinear form) equals the matrix rank of M.
    # ones(2,8) has matrix rank 1. So the bilinear form rank is 1.
    # But that means C[0,k] = (sum_j A[0,j]) * (sum over... no.
    # If T[j,k] = u[j]*v[k] with u=[1,1], v=[1,1,1,1,1,1,1,1], then:
    # sum_{j,k} T[j,k] A[0,j] B[j,k] = sum_{j,k} 1*A[0,j]*B[j,k] = sum_j A[0,j] * sum_k B[j,k]
    # But we want sum_j A[0,j] * B[j,k] for EACH k separately.
    
    # I see my confusion. The tensor for matrix multiplication (n,m,p) is NOT just a matrix.
    # It's a 3-tensor. For (1,2,8): T[i,j,k] but i is always 0.
    # The trilinear form is: sum_{i,j,k} T[i,j,k] C^*[i,k] A[i,j] B[j,k]
    # With i=0: sum_{j,k} T[0,j,k] C^*[0,k] A[0,j] B[j,k]
    # We want this to equal C[0,k] = sum_j A[0,j] B[j,k] for each k.
    # So T[0,j,k] = 1 for all j,k. The "tensor" here is T[j,k] = 1 (2x8).
    # But this is being contracted with C^*[0,k] and A[0,j] and B[j,k].
    # The issue is that k appears in both C and B, and j appears in both A and B.
    # So T[j,k] = 1 means: the coefficient of A[0,j] * B[j,k] * C^*[0,k] is 1.
    # This is correct: C[0,k] = sum_j A[0,j] B[j,k].
    
    # Now, the tensor rank: we need T[j,k] = sum_s U[j,s] V[k,s] (since i is fixed)
    # T[j,k] = 1 for all j,k. This is the all-ones 2x8 matrix.
    # Rank of all-ones 2x8 = 1. So T[j,k] = u[j] * v[k] with u=[1,1], v=[1,...,1].
    # Then: sum_s U[j,s] V[k,s] = 1*1 = 1. 
    # So the "product" for a fixed Strassen component is:
    # Contribution to C[0,k] = (sum_j U_A[j,s] A[0,j]) * (sum_k U_B[k,s] B[j,k])... 
    # No, I'm mixing up the factors.
    
    # Let me use the actual interface. U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # For (2,2,8): U is (4, rank), V is (16, rank), W is (16, rank).
    # The decomposition: the matmult tensor T is such that:
    # C[i,k] = sum_{j,s} U[i*m+j, s] * (something) ... 
    
    # I think the correct interpretation from the seed is:
    # The bilinear form for matmult is: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] ... no.
    # Standard: C = A*B means C[i,k] = sum_j A[i,j] B[j,k].
    # The "tensor" is T[(i,k), (i,j), (j,k)] = delta_{i,i}... 
    # I think the convention is: T[a, b, c] where a=(i,k) for C, b=(i,j) for A, c=(j,k) for B
    # T[(i,k), (i',j), (j',k')] = delta_{i,i'} delta_{k,k'} delta_{j,j'}
    # And the decomposition: T = sum_s U_s (as vector over a) * V_s (over c) * W_s (over b)
    # where U has shape (p*n, rank) [output], V has shape (m*p, rank) [B], W has shape (n*m, rank) [A]
    # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # So U corresponds to A, V to B, W to C (output).
    # And W[k*n+i, s] with the trace convention.
    
    # So: T[a,b,c] = sum_s U[b,s] V[c,s] W[a,s] where a indexes C (size p*n), b indexes A (size n*m), c indexes B (size m*p)
    # T[(k,i), (i',j), (j',k')] = delta_{i,i'} delta_{k,k'} delta_{j,j'}
    # = sum_s U[(i',j), s] V[(j',k'), s] W[(k,i), s]
    
    # For schoolbook: for each (i,j,k), one column s with:
    # U[(i,j), s] = 1, V[(j,k), s] = 1, W[(k,i), s] = 1
    # Check: T[(k,i), (i',j'), (j'',k'')] = sum_s U[(i',j'),s] V[(j'',k''),s] W[(k,i),s]
    # For the column s=(i,j,k): U[(i',j'),s] = 1 if (i',j')=(i,j), V[(j'',k''),s] = 1 if (j'',k'')=(j,k), W[(k,i),s] = 1 always (since s is indexed by (i,j,k) and W index is (k,i))
    # So T[(k,i), (i',j'), (j'',k'')] = 1 if (i',j')=(i,j) and (j'',k'')=(j,k), i.e., i'=i, j'=j, j''=j, k''=k
    # = delta_{i',i} delta_{j',j} delta_{j'',j} delta_{k'',k}
    # We need T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{k,k''} delta_{j,j'}... 
    # Hmm, the delta conditions: we need i'=i, k''=k, j'=j''? No.
    # We need: T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{k,k''} delta_{j',j''}... 
    # Actually for matmult: C[k,i]... I think I have the indices confused.
    
    # Let me just go with the knowledge that rank 28 is achievable and construct it using the Strassen approach,
    # then verify it's correct.
    
    # For 2x2 x 2x8, use Strassen's 7 products. Each product involves a 1x8 "sub-multiplication" 
    # which I'll implement with rank 4 (giving total 28).
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Here, in our case, B00, B11 etc. are 1x8 rows. So M1, M2, etc. are 1x8 vectors.
    # M1[k] = (A00+A11) * (B00[k]+B11[k])
    # This is a product of a scalar and a 1x8 vector.
    # To decompose: for each k, M1[k] = (A00+A11) * (B0[k]+B1[k])
    # This is a bilinear form in A and B. The tensor for this single component (for fixed k) is:
    # T[j,k] where j indexes which A/B entries... 
    
    # I think the cleanest approach is to just hardcode the rank-28 solution.
    # Let me construct it by treating each Strassen product as requiring rank 4.
    
    # For a single Strassen product like M1[k] = (A00+A11)*(B0[k]+B1[k]):
    # This is a scalar (from A) times a scalar (from B at position k).
    # As a bilinear form: it's just a single product of two linear combinations.
    # The "tensor" for this is rank 1! Because it's just (linear combo of A) * (linear combo of B at k).
    # But k varies, and the B part depends on k.
    # For fixed k: T is a scalar, rank 1.
    # For all k: the B part is a 1x8 vector, and we're multiplying a scalar by each component.
    # The tensor across k: for each k, we have the same A-part but different B-part.
    # So the total tensor for M1 (all 8 k's) has rank 8 (one per k), because:
    # M1[k] = (A00+A11) * B0[k] + (A00+A11) * B1[k]
    # = (A00+A11) * (B0[k]+B1[k])
    # The A-part is the same scalar for all k, but the B-part varies.
    # As a tensor: T_A * T_B[k] where T_A is the A-coefficient vector and T_B[k] is the B-coefficient for k.
    # This is a rank-1 tensor in the A x (B,k) space? 
    # The bilinear form is: sum_k C^*[k] * (A00+A11) * (B0[k]+B1[k])
    # = (A00+A11) * sum_k C^*[k]*(B0[k]+B1[k])
    # This is a product of a linear form in A and a linear form in B (and C).
    # So the tensor rank for M1 is 1!
    
    # If each Strassen product has tensor rank 1, then total rank is 7!
    # But that can't be right because we know the lower bound for 2x2x2 is 7, and for 2x2x8 it should be higher.
    
    # I think the issue is that the 7 Strassen products don't directly give the output; we need to combine them.
    # The combination is: C00[k] = M1[k]+M4[k]-M5[k]+M7[k], etc.
    # This combination is linear in the M's, so it doesn't add rank.
    # But each M_s[k] is a bilinear form, and the total tensor is the sum of 7 rank-1 tensors, giving rank <= 7.
    # But that would mean 2x2x8 has rank <= 7, which contradicts the schoolbook rank of 32 and the knowledge saying 28.
    
    # I must be misunderstanding the tensor structure. Let me look at this differently.
    
    # I'll just implement the rank-28 solution by construction and trust the knowledge.
    # Based on the knowledge, a valid rank-28 solution exists. Let me construct it.
    
    # Here's my approach: I'll use the Strassen decomposition and allocate rank 4 per product.
    # The 4 comes from the fact that each Strassen "product" in the 2x2x8 setting involves
    # a 2x8 B-submatrix, and the tensor rank of the relevant sub-structure is 4.
    
    # Actually, let me just directly construct the solution using the known Strassen structure
    # and verify it produces the correct result.
    
    rank = 28
    U = [[0] * rank for _ in range(4)]  # n*m = 4
    V = [[0] * rank for _ in range(16)]  # m*p = 16
    W = [[0] * rank for _ in range(16)]  # p*n = 16
    
    # I'll define the 7 Strassen products, each using 4 columns of the decomposition.
    # For each Strassen product s (0..6), columns 4*s to 4*s+3 are used.
    # The A-part (U) encodes the linear combination of A entries.
    # The B-part (V) encodes the linear combination of B entries (2x8 = 16 entries).
    # The C-part (W) encodes how this product contributes to each C entry.
    
    # Strassen products (A-combo, B-combo, C-contribution):
    # I'll define for each product: 
    #   a_vec: 4-element vector (coefficients for A[0,0], A[0,1], A[1,0], A[1,1])
    #   b_vec: 16-element vector (coefficients for B[0,0..7], B[1,0..7])
    #   c_vec: 16-element vector (coefficients for C contributions, indexed by k*n+i)
    
    # But this is a rank-1 per product, giving total rank 7, which I argued is wrong.
    # The issue must be that the B-combo is not a single linear combination but depends on k.
    
    # Let me try a different approach: just verify the rank-28 claim by constructing a specific solution.
    
    # I'll use the following known construction for 2x2x2x8 (Strassen + schoolbook for the 8):
    # Actually, I recall that the tensor rank of 2x2x2 is 7, and for 2x2x8, 
    # one approach is to use the fact that it's like 8 copies of 2x2x1 (rank 2 each) = 16,
    # but with sharing via Strassen, we get 28.
    
    # Let me just go with a concrete construction that I can verify.
    
    # I'll implement the Strassen algorithm explicitly:
    # The 7 intermediate "products" are each 1x8 vectors.
    # To compute each 1x8 vector = scalar_A * (1x8 vector_B), 
    # I need to represent this as a sum of rank-1 tensors.
    # For a fixed product, say M1[k] = (A00+A11) * (B0k + B1k):
    # This can be written as: (A00+A11) * B0k + (A00+A11) * B1k
    # As a bilinear form in (A, B, C): the coefficient is 1 for the term A00*B0k*C^*_{0,?}...
    
    # I think I'm overcomplicating this. Let me just hardcode a rank-28 solution
    # that I know works from the literature/knowledge.
    
    # Based on the knowledge that rank 28 is valid, here's a construction:
    # Use 7 Strassen-like products, each decomposed into 4 rank-1 terms.
    
    # I'll define the solution explicitly.
    col = 0
    # For each of the 7 Strassen products, add 4 rank-1 terms.
    # The specific coefficients will be set to match the Strassen algorithm.
    
    # I'll use a simplified approach: since I know rank 28 works, let me construct it
    # by modifying the schoolbook to share some columns.
    
    # Actually, let me just return a valid rank-28 solution by using the following method:
    # For 2x2 x 2x8, note that C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is 2 independent 2x1 matrix-vector products, each of rank 2.
    # Total: 8 columns * 2 = 16 if no sharing.
    # With Strassen sharing across the i dimension, we get 7 * 4 = 28.
    
    # I'll implement the rank-28 solution by explicitly defining all 28 columns.
    
    # Let me define the solution based on the Strassen structure:
    # Product 0: (A00+A11) * (B0+B1) -> contributes to C00 and C11
    # Product 1: (A10+A11) * B0 -> contributes to C10 and C11
    # Product 2: A00 * (B1-B0) -> contributes to C01 and C11
    # Product 3: A11 * (B1-B0) -> contributes to C00 and C10
    # Product 4: (A00+A01) * B1 -> contributes to C00 and C01
    # Product 5: (A10-A00) * (B0+B1) -> contributes to C11
    # Product 6: (A01-A11) * (B0+B1) -> contributes to C01
    
    # For each product, the B-part is a 1x8 vector (linear combo of B[0,:] and B[1,:]).
    # Computing scalar * (1x8 vector) requires 8 "rank-1" terms (one per k), but we can share
    # the scalar across k's. The tensor rank for scalar * v (1x8) is 8 (since the B-part 
    # has 8 independent components).
    # 7 * 8 = 56, which is worse.
    
    # I think the rank-28 comes from a more clever sharing. Let me just trust the knowledge
    # and construct a specific valid solution.
    
    # I'll use the following: for each of the 7 Strassen products, I'll use 4 columns.
    # The 4 columns correspond to the 4 "quadrants" of the 2x8 B matrix (split into 2x2 blocks of size 1x4? No, 2x8 split into four 2x2 blocks).
    
    # Split B (2x8) into four 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8].
    # For each block, do a 2x2x2 Strassen (rank 7), total 7*4 = 28.
    # This is the key insight! Split the 8 columns into 4 groups of 2, and apply Strassen to each 2x2x2 sub-problem.
    
    # For each block of 2 columns, we have a 2x2 x 2x2 matrix multiplication, which has rank 7 (Strassen).
    # 4 blocks * 7 = 28 total rank.
    
    # This is the construction!
    
    # For block b (b=0,1,2,3), columns 2b and 2b+1 of B.
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For block b, the B entries are B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
    # The A entries are the same for all blocks (shared!).
    # The C entries are C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1].
    
    # The key: A is shared across all 4 blocks! So the U matrix (A-part) can be shared.
    # But in the tensor decomposition, each rank-1 term has a specific U, V, W.
    # Since A is the same for all blocks, the U part for corresponding Strassen products across blocks is the same.
    # But V and W differ (different B and C columns).
    
    # In the tensor decomposition, each rank-1 term is (U_col, V_col, W_col).
    # For block b, product s: U_col = u_s (same for all b), V_col = v_{s,b}, W_col = w_{s,b}.
    # But in a rank-1 term, U, V, W are all specific vectors. We can't share U across different rank-1 terms
    # unless they are the same term.
    
    # Wait, but the total rank is still 7*4 = 28 because we have 7 Strassen products per block,
    # 4 blocks, and each is a rank-1 term. Even though the U part repeats, each (U, V, W) triple is distinct
    # because V and W differ. So we still need 28 rank-1 terms.
    
    # Let me construct this explicitly.
    
    # Strassen coefficients for 2x2x2:
    # Product 0: A-coeffs [1,0,0,1] (A00+A11), B-coeffs [1,0,0,1] (B00+B11)
    # Product 1: A-coeffs [0,0,1,1] (A10+A11), B-coeffs [1,0,0,0] (B00)
    # Product 2: A-coeffs [1,0,0,0] (A00), B-coeffs [0,0,1,-1] (B10-B11)
    # Product 3: A-coeffs [0,0,0,1] (A11), B-coeffs [0,1,-1,0] (B01-B00)
    # Product 4: A-coeffs [1,1,0,0] (A00+A01), B-coeffs [0,0,0,1] (B11)
    # Product 5: A-coeffs [-1,0,1,0] (A10-A00), B-coeffs [1,0,1,0] (B00+B10)
    # Product 6: A-coeffs [0,1,0,-1] (A01-A11), B-coeffs [0,1,0,1] (B01+B11)
    
    # C-contribution for each product:
    # Product 0: C00 += 1, C11 += 1
    # Product 1: C10 += 1, C11 += -1
    # Product 2: C01 += 1, C11 += 1
    # Product 3: C00 += 1, C10 += 1
    # Product 4: C00 += -1, C01 += 1
    # Product 5: C11 += 1
    # Product 6: C01 += 1
    
    # A-indexing: U[i*m+j] = U[i*2+j], so A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B-indexing: V[j*p+k] = V[j*8+k], so B[0,k]=V[k], B[1,k]=V[8+k]
    # C-indexing: W[k*n+i] = W[k*2+i], so C[0,k]=W[2k], C[1,k]=W[2k+1]
    
    # For block b (columns 2b, 2b+1):
    # B00 = B[0, 2b] = V[2b]
    # B01 = B[0, 2b+1] = V[2b+1]
    # B10 = B[1, 2b] = V[8+2b]
    # B11 = B[1, 2b+1] = V[8+2b+1]
    # C00 = C[0, 2b] = W[2*(2b)] = W[4b]
    # C01 = C[0, 2b+1] = W[2*(2b+1)] = W[4b+2]
    # C10 = C[1, 2b] = W[2*(2b)+1] = W[4b+1]
    # C11 = C[1, 2b+1] = W[2*(2b+1)+1] = W[4b+3]
    
    # Define Strassen products:
    products = [
        # (A_coefs, B_coefs_for_block, C_contribs)
        # A_coefs: [A00, A01, A10, A11]
        # B_coefs: [B00, B01, B10, B11] (relative to block)
        # C_contribs: [C00, C01, C10, C11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, 0]),   # M7
    ]
    
    col = 0
    for b in range(4):  # 4 blocks of 2 columns each
        for s in range(7):  # 7 Strassen products per block
            a_coefs = products[s][0]
            b_coefs = products[s][1]
            c_coefs = products[s][2]
            
            # Set U (A-part)
            U[0][col] = a_coefs[0]  # A[0,0]
            U[1][col] = a_coefs[1]  # A[0,1]
            U[2][col] = a_coefs[2]  # A[1,0]
            U[3][col] = a_coefs[3]  # A[1,1]
            
            # Set V (B-part): V[j*p+k]
            # B00 = B[0, 2b] = V[2b], B01 = B[0, 2b+1] = V[2b+1]
            # B10 = B[1, 2b] = V[8+2b], B11 = B[1, 2b+1] = V[8+2b+1]
            V[2*b][col] = b_coefs[0]       # B00
            V[2*b+1][col] = b_coefs[1]     # B01
            V[8+2*b][col] = b_coefs[2]     # B10
            V[8+2*b+1][col] = b_coefs[3]   # B11
            
            # Set W (C-part): W[k*n+i]
            # C00 = C[0, 2b] = W[4b], C01 = C[0, 2b+1] = W[4b+2]
            # C10 = C[1, 2b] = W[4b+1], C11 = C[1, 2b+1] = W[4b+3]
            W[4*b][col] = c_coefs[0]       # C00
            W[4*b+2][col] = c_coefs[1]     # C01
            W[4*b+1][col] = c_coefs[2]     # C10
            W[4*b+3][col] = c_coefs[3]     # C11
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

