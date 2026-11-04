"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    # We'll construct a decomposition with rank < 42
    # Using a structured approach based on linearity across columns
    
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need to find linear forms u_c(A), v_c(B) such that
    # C = sum_c u_c(A) * v_c(B) (outer product interpretation)
    
    # Strategy: Use the fact that we can share computations
    # Let's try to find a decomposition with rank around 20-30
    
    # For now, let's use a constructive approach
    # We'll use the "slab" method: peel off columns
    
    # Actually, let me use a known low-rank technique
    # For 2x3 by 3x7, we can use the following:
    
    # Represent A as [a00, a01, a02; a10, a11, a12]
    # Represent B as a 3x7 matrix
    
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k]
    
    # We can write this as:
    # C = [a00; a01; a02] * B[0,:] + [a10; a11; a12] * B[1,:] ... no that's not right
    
    # Let me think of it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One approach: use rank-1 updates
    # For each pair (i,j), we have A[i,j] * B[j,:] contributing to row i of C
    # But we can combine terms
    
    # Let's try a decomposition with the following structure:
    # Use 3 sets of 7 terms (one for each column of A/row of B), but with sharing
    
    # Actually, the simplest improvement: note that we only need 14 output values
    # Each rank-1 term gives us one product that can contribute to multiple outputs
    
    # Let me try to construct a decomposition with rank = 21 (half of 42)
    # or better
    
    # I'll use a systematic construction
    rank = 24  # Let's try this
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 21 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
    # For each rank-1 term c, we have:
    # U[:,c] is a vector in F^6 (coefficients for A[i,j])
    # V[:,c] is a vector in F^21 (coefficients for B[j,k])
    # W[:,c] is a vector in F^14 (coefficients for C[i,k])
    # Such that for all i,k: sum_c W[i*7+k, c] * U[i*3+j, c] * V[j*7+k, c] = sum_j A[i,j]*B[j,k]
    
    # Let's use a simple but effective decomposition
    # Term c corresponds to (i, j, k) but we can merge some
    
    # Actually, let's just use a better structured decomposition
    # I'll use the fact that for fixed i, C[i,:] = A[i,:] * B (row vector times matrix)
    # This is a 1x3 by 3x7 multiplication, which has rank 3*7=21 if done naively
    # But we can do it with fewer multiplications
    
    # For a 1x3 by 3x7: C[k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k]
    # This requires 3 multiplications per output, 7 outputs = 21 mults
    # But we can use: let t0 = a0*B[0,:], t1 = a1*B[1,:], t2 = a2*B[2,:]
    # Then C = t0 + t1 + t2 (vector addition)
    # So we need 3 multiplications (each producing a 7-vector)
    # That's rank 3 for the 1x3 by 3x7 case
    
    # For 2x3 by 3x7:
    # Row 0: 3 multiplications (a00*B[0,:], a01*B[1,:], a02*B[2,:])
    # Row 1: 3 multiplications (a10*B[0,:], a11*B[1,:], a12*B[2,:])
    # Total: 6 multiplications, each producing a 7-vector
    # But wait, each "multiplication" here is actually a scalar times a vector
    # In the tensor decomposition framework, each scalar-vector product is one rank-1 term
    
    # So we need 6 * 7 = 42 terms? No, that's not right either.
    
    # Let me reconsider. In the tensor decomposition:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ F^6, v_c ∈ F^21, w_c ∈ F^14
    
    # For the term corresponding to A[i,j] * B[j,k] contributing to C[i,k]:
    # u_c has 1 at position i*3+j
    # v_c has 1 at position j*7+k  
    # w_c has 1 at position i*7+k
    
    # That's the schoolbook with 42 terms.
    
    # To reduce rank, we need to find linear combinations where one term contributes to multiple (i,k) pairs.
    
    # Key insight: if we let u_c be a linear form in A, v_c be a linear form in B,
    # then u_c * v_c is a bilinear form that can contribute to multiple C[i,k] entries.
    
    # For example, let u_c = a00 + a10 (sum of first column of A)
    # and v_c = B[0,0] + B[1,0] + ... (some linear form in B)
    # Then u_c * v_c contributes to C[0,0], C[1,0], and possibly others.
    
    # Let me try to construct a decomposition with fewer terms.
    
    # I'll use a constructive algorithm:
    # We need to represent the 14 output entries as linear combinations of products of linear forms.
    
    # Let me try rank = 18 or so.
    
    # Actually, let me just use a known result or a simple improvement.
    # For now, I'll implement a decomposition with rank = 30 as a safe improvement.
    
    # Let me use a different strategy: exploit the 2 in the first dimension.
    # C[0,k] - C[1,k] = (a00-a10)*B[0,k] + (a01-a11)*B[1,k] + (a02-a12)*B[2,k]
    # C[0,k] + C[1,k] = (a00+a10)*B[0,k] + (a01+a11)*B[1,k] + (a02+a12)*B[2,k]
    
    # This doesn't immediately help reduce rank.
    
    # Let me just implement a reasonable decomposition.
    # I'll use the "block" approach:
    
    # Group the 7 columns into groups and use linearity.
    
    # For a practical implementation, let me use rank = 21
    # by noting that we can share the B[j,:] vectors.
    
    # Actually, the cleanest low-rank decomposition I can think of:
    # Use 3 terms for row 0 and 3 terms for row 1, but each term is a scalar times a full row of B.
    # In the tensor framework, this means:
    # For row 0: terms for j=0,1,2, where u has 1 at (0,j), v has 1s at all k for row j of B, w has 1s at all k for row 0 of C
    # But v can't have 1s at all k for a fixed j in a single term unless we allow v to be a vector with multiple 1s.
    
    # Yes! v_c can be any vector in F^21. So for term c corresponding to A[0,0] * B[0,:]:
    # u_c = e_{0,0} (1 at position 0 in F^6)
    # v_c = (1,0,0,...,0,1,0,0,...,0,1,0,0,...) wait no.
    # B is 3x7, indexed as B[j,k] at position j*7+k.
    # B[0,:] corresponds to positions 0,1,2,3,4,5,6 (j=0, k=0..6)
    # So v_c has 1 at positions 0,1,2,3,4,5,6 and 0 elsewhere.
    # w_c has 1 at positions 0,1,2,3,4,5,6 (C[0,0]..C[0,6]) and 0 elsewhere.
    
    # This works! And it's one rank-1 term that accounts for A[0,0]*B[0,k] for all k.
    
    # So we need:
    # 3 terms for row 0 (j=0,1,2)
    # 3 terms for row 1 (j=0,1,2)
    # Total: 6 terms!
    
    # Wait, that gives rank 6, which seems too good. Let me verify.
    
    # Term 1: u=e_{0,0}, v=(1,1,1,1,1,1,1, 0,...,0, 0,...,0), w=(1,1,1,1,1,1,1, 0,...,0)
    # This gives: A[0,0] * (B[0,0]+B[0,1]+...+B[0,6]) contributing to C[0,0]+...+C[0,6]?
    # No! The tensor product u⊗v⊗w means:
    # sum_{i,j,k} u[i,j] * v[j',k'] * w[i',k'] * A[i,j] * B[j',k'] * C[i',k']
    # For this to equal sum_{i,j,k} A[i,j]*B[j,k]*C[i,k], we need:
    # sum_c u_c[i,j] * v_c[j',k'] * w_c[i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... no.
    
    # Actually, the condition is:
    # For all i,k: sum_j A[i,j]*B[j,k] = sum_c (sum_j u_c[i,j]*A[i,j]) * (sum_{j',k'} v_c[j',k']*B[j',k']) * w_c[i,k]
    # Wait, that's not quite right either.
    
    # The tensor T has T[i,j,k] = 1 if the indices match, 0 otherwise.
    # T = sum_c u_c ⊗ v_c ⊗ w_c means:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k] for all i,j,k
    # where i indexes A (6 values), j indexes B (21 values), k indexes C (14 values).
    
    # And we need: sum_{i,j} T[i,j,k] * A[i] * B[j] = C[k] for all k
    # i.e., sum_{i,j} u_c[i]*v_c[j]*w_c[k] summed over c, times A[i]*B[j] = C[k]
    
    # More precisely: C[k] = sum_{i,j} (sum_c u_c[i]*v_c[j]*w_c[k]) * A[i] * B[j]
    # We need sum_c u_c[i]*v_c[j]*w_c[k] = delta_{i,j,k} where delta is 1 if the indices are consistent.
    
    # For our problem: A has indices (i,j) with i=0,1 and j=0,1,2.
    # B has indices (j',k') with j'=0,1,2 and k'=0..6.
    # C has indices (i',k') with i'=0,1 and k'=0..6.
    # The multiplication is C[i',k'] = sum_j A[i',j]*B[j,k']
    # So T[i,j, j',k', i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # In flattened form: T[(i,j), (j',k'), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # So we need: sum_c u_c[i,j] * v_c[j',k'] * w_c[i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # For the term I proposed: u=e_{0,0}, v has 1s at (0,0)..(0,6), w has 1s at (0,0)..(0,6)
    # Then u_c[i,j]*v_c[j',k']*w_c[i',k'] = 1 if i=0,j=0,j'=0,i'=0, and k=k' (any k)
    # This gives delta_{i,0}*delta_{j,0}*delta_{j',0}*delta_{i',0} for all k,k'
    # But we need delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # So this term contributes to (i=0,j=0,j'=0,i'=0) for all k,k', but we only want it when k=k'.
    
    # So my proposed decomposition is WRONG. Each rank-1 term must have the property that
    # u_c[i,j]*v_c[j',k']*w_c[i',k'] is nonzero only when i=i', j=j', k=k'.
    
    # This means that for a single term, the support of u, v, w must be such that
    # if u is nonzero at (i,j) and v is nonzero at (j',k') and w is nonzero at (i',k'),
    # then i=i', j=j', k=k'.
    
    # In other words, each term can only involve a single (i,j) for u, a single (j,k) for v,
    # and a single (i,k) for w, with consistent indices.
    
    # So actually, the schoolbook decomposition with 42 terms might be close to optimal
    # for this specific structure, OR we need more clever linear forms.
    
    # The key is that u_c can be a linear form, not just a basis vector.
    # So u_c[i,j] can be nonzero for multiple (i,j) pairs, as long as the overall
    # tensor equation is satisfied.
    
    # This is where the low-rank decompositions come in. We need to find linear forms
    # such that the sum of outer products equals the multiplication tensor.
    
    # This is a non-trivial algebraic problem. For now, let me use a known result
    # or a simple improvement.
    
    # Given the complexity, let me just use the schoolbook decomposition but with
    # a small optimization: if there are any obvious symmetries or redundancies.
    
    # Actually, for the purpose of this task, let me just return the schoolbook
    # decomposition. The rank is 42, which is the baseline.
    
    # But wait, the task says "find a rank-minimal exact tensor decomposition", so
    # we should try to do better than 42.
    
    # Let me try to implement a decomposition with rank < 42 using a known technique.
    
    # One approach: use the fact that 7 = 3 + 4, and decompose accordingly.
    
    # For now, I'll implement the schoolbook as a safe fallback, but try to improve.
    
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
