"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    # Target tensor
    Tt = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tt[i*m+j, j*p+k, k*n+i] = 1.0
    return np.max(np.abs(T - Tt)) < tol

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to use known low-rank constructions
    # For 3x3x3, rank 23 is minimal (Lader)
    # For 3x3x7, let's try to construct a decomposition
    
    # Approach: Use the structure that C[i,k] = sum_j A[i,j]*B[j,k]
    # We can group terms. Let's try a direct construction.
    
    # Schoolbook is rank 63. Let's see if we can do better.
    # 
    # Key insight: For the 3x3 part, we know rank 23 works.
    # The 3x3x3 Lader decomposition can be adapted.
    #
    # Let me try: treat the p=7 dimension and see if we can share
    # the A factors across different k values.
    
    # Actually, let's try a simple construction:
    # For each i (row of C), C[i,:] = A[i,:] * B (row vector times matrix)
    # This is a 1x3 * 3x7 = 1x7 multiplication
    # The rank of 1x3 x 3x7 multiplication tensor is 3 (since it's just 3 scalar multiplications summed)
    # Wait no: C[i,k] = sum_j A[i,j]*B[j,k], for fixed i this is a bilinear map
    # (A[i,0],A[i,1],A[i,2]) × (B[:,k] for all k) -> C[i,k]
    # The rank of this sub-tensor is 3 (each j gives one term)
    # So total rank ≤ 3*3 = 9? No, that's not right because the B factors are shared.
    
    # Let me just try the schoolbook and see if I can find a better one via search.
    
    # Given time constraints, let me try a constructive approach:
    # Use the fact that for 3x3x3, rank 23 is optimal.
    # For 3x3x7, one can use: rank(3x3x7) ≤ rank(3x3x3) + 3*rank(3x3x1)*4/3... 
    # This is getting complicated. Let me just try to implement a known construction.
    
    # Lader's 3x3x3 rank-23 decomposition:
    # The 23 terms are:
    # 9 diagonal terms: (i,i,i) for i=0,1,2 -> 3 terms
    # Wait, I need to recall the exact construction.
    
    # Let me try a different approach: use the formula
    # A*B = (A+B)^2/4 + (A-B)^2/4 - ... no that's for commutative.
    
    # For non-commutative 3x3, the rank-23 decomposition is specific.
    # Let me just use schoolbook for now and try to improve.
    
    # Actually, let me try a simple improvement: 
    # Group by i. For each i, C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a 1x3 * 3x7 multiplication.
    # The rank of 1x3 * 3x7 is 3 (not 7, because we only need 3 products summed).
    # Wait: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This is 3 terms, so rank 3 for fixed i.
    # Total: 3 * 3 = 9? But the B factors are the same across i!
    # So we can share: total rank = 3 (for j=0) + 3 (for j=1) + 3 (for j=2) = 9?
    # No: for j=0, we have terms A[i,0]*B[0,k] for all i,k.
    # That's a 3x1 * 1x7 = 3x7 outer product structure, rank 3.
    # For j=1: another rank 3.
    # For j=2: another rank 3.
    # Total rank = 9.
    
    # Wait, that can't be right. Let me verify: 
    # Term for j=0: U has A[i,0] for i=0,1,2 (3 entries), V has B[0,k] for k=0..6 (7 entries), W has 1 for (k,i) pairs... 
    # No, each rank-1 term is u ⊗ v ⊗ w where u ∈ R^9, v ∈ R^21, w ∈ R^21.
    
    # For j=0: u[i*3+0] = A[i,0] coefficient, v[0*7+k] = B[0,k] coefficient, w[k*3+i] = 1
    # But u, v, w are fixed vectors, not dependent on the input matrices.
    # The decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[(i,j),(j,k),(k,i)] = 1
    
    # For the j=0 group: T[(i,0),(0,k),(k,i)] = 1 for all i,k
    # This is a rank-1 tensor? u_r[i*3+0]=1, v_r[0*7+k]=1, w_r[k*3+i]=1
    # But u_r is in R^9, so u_r has entries at positions 0,3,6 (for i=0,1,2, j=0)
    # v_r has entries at positions 0,1,2,3,4,5,6 (for j=0, k=0..6)
    # w_r has entries at positions k*3+i for all k,i
    # This is NOT rank 1 because w_r would need to be 1 at all 21 positions,
    # but u_r and v_r are also not single-entry vectors.
    
    # Actually, the sub-tensor for fixed j IS rank 1 in the appropriate sense:
    # T_j[(i,0),(0,k),(k,i)] = 1 = u_i * v_k * w_{k,i} where u_i=1, v_k=1, w_{k,i}=1
    # So yes, each j gives a rank-1 term!
    # Total rank = 3 (one for each j=0,1,2).
    
    # Wait, that means rank 3x3x7 ≤ 3?? That can't be right.
    # Let me re-examine. The tensor T has T[(i,j),(j,k),(k,i)] = 1.
    # For fixed j=0: T[(i,0),(0,k),(k,i)] = 1 for all i∈{0,1,2}, k∈{0,...,6}
    # This is a 3x7x21 sub-tensor (well, 3*7=21 non-zero entries in the 9x21x21 tensor)
    # Can this be written as a single rank-1 tensor u⊗v⊗w?
    # u would need u[i*3+0] = a_i, v[0*7+k] = b_k, w[k*3+i] = c_{k,i}
    # And a_i * b_k * c_{k,i} = 1 for all i,k
    # This is possible: set a_i=1, b_k=1, c_{k,i}=1 for all i,k
    # So yes! The j=0 sub-tensor IS rank 1.
    # Similarly for j=1 and j=2.
    # Total rank = 3.
    
    # But wait, that contradicts the lower bound rank ≥ max(nm, mp, pn) = max(9,21,21) = 21.
    # Where is my error?
    
    # The error: the rank-1 tensor u⊗v⊗w contributes to ALL entries, not just the j=0 ones.
    # If I use u with non-zero entries at positions 0,3,6 (j=0) and v with non-zero at 0-6 (j=0)
    # and w with non-zero at all k*3+i positions, then this rank-1 tensor has non-zero entries
    # at (i*3+0, 0*7+k, k*3+i) for all i,k, which is exactly what we want for j=0.
    # But it also has ZERO entries everywhere else, which is fine.
    # So the sum of 3 such rank-1 tensors (for j=0,1,2) gives the full tensor.
    # Rank = 3?!
    
    # This must be wrong. Let me check with a small case: 2x2x2.
    # T[(i,j),(j,k),(k,i)] = 1
    # For j=0: T[(i,0),(0,k),(k,i)] = 1 for i,k∈{0,1}
    # Rank-1: u[i*2+0]*v[0*2+k]*w[k*2+i] = 1
    # u[0]=a0, u[2]=a1, v[0]=b0, v[1]=b1, w[0]=c00, w[1]=c10, w[2]=c01, w[3]=c11
    # a0*b0*c00=1, a0*b1*c10=1, a1*b0*c01=1, a1*b1*c11=1
    # Set a0=a1=1, b0=b1=1, c00=c10=c01=c11=1. Works!
    # Similarly for j=1.
    # Total rank 2 for 2x2x2? But we know rank(2x2x2) = 7 (Strassen).
    
    # So my reasoning is flawed. The issue is that the rank-1 tensor u⊗v⊗w 
    # has u in R^{nm}, v in R^{mp}, w in R^{pn}.
    # For the j=0 term: u has non-zeros at positions i*m+0 for i=0..n-1
    # v has non-zeros at positions 0*p+k for k=0..p-1
    # w has non-zeros at positions k*n+i for all k,i
    # The product u⊗v⊗w has non-zeros at (i*m+0, 0*p+k, k*n+i) for all i,k
    # This is exactly the j=0 slice. And it's a valid rank-1 tensor.
    # So rank(3x3x7) ≤ 3??
    
    # This is clearly wrong. Let me check the definition of tensor rank again.
    # The tensor T ∈ U⊗V⊗W has rank r if T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    # where u_s ∈ U, v_s ∈ V, w_s ∈ W.
    # Each term u_s⊗v_s⊗w_s is a tensor of rank 1, meaning it can be written as
    # (u_s[i] * v_s[j] * w_s[k]) for indices i,j,k.
    
    # For the j=0 slice: T[(i,0),(0,k),(k,i)] = 1
    # Let u_s[i*m+0] = 1 for i=0,1,2, and 0 otherwise.
    # v_s[0*p+k] = 1 for k=0..6, and 0 otherwise.
    # w_s[k*n+i] = 1 for all k=0..6, i=0,1,2.
    # Then (u_s⊗v_s⊗w_s)[(i,0),(0,k),(k,i)] = 1*1*1 = 1. ✓
    # And (u_s⊗v_s⊗w_s)[(i,j),(j',k),(k',i')] = 0 unless j=0, j'=0, and k'=k, i'=i.
    # Wait, the indices are: first index is (i,j) flattened to i*m+j, second is (j,k) to j*p+k, third is (k,i) to k*n+i.
    # So (u_s⊗v_s⊗w_s)[a,b,c] = u_s[a]*v_s[b]*w_s[c]
    # For a=i*m+0, b=0*p+k, c=k*n+i: u_s[a]=1, v_s[b]=1, w_s[c]=1, product=1. ✓
    # For a=i*m+1 (j=1): u_s[a]=0, product=0. ✓
    # So this IS a valid rank-1 tensor that captures the j=0 slice.
    
    # So rank(3x3x7) ≤ 3. But the lower bound says rank ≥ 21. Contradiction!
    
    # I must be misunderstanding the tensor. Let me re-read the problem.
    # "W[k*n+i,r]" - so the third factor's index is k*n+i, where k is the column of B and row of C,
    # and i is the row of A and C.
    # The tensor is T[(i,j), (j,k), (k,i)] = 1, representing C[k,i] += A[i,j]*B[j,k]
    # Wait, C is p×n = 7×3, so C[k,i] where k=0..6, i=0..2.
    # A is n×m = 3×3, B is m×p = 3×7.
    # C[k,i] = sum_j B[j,k]*A[i,j] (or A[i,j]*B[j,k], same thing)
    
    # So the tensor T has T[(i,j), (j,k), (k,i)] = 1.
    # This is what I had. And my argument seems correct that rank ≤ 3.
    # But that's impossible given known results.
    
    # OH WAIT. I think I see the issue. The rank-1 tensor u⊗v⊗w where
    # u has 3 non-zero entries, v has 7 non-zero entries, w has 21 non-zero entries
    # is NOT a valid "term" in the tensor decomposition in the way I'm thinking.
    # Actually it IS valid. A rank-1 tensor is any tensor of the form u⊗v⊗w.
    # The number of non-zero entries doesn't matter.
    
    # Let me just verify numerically for a tiny case: 1x1x1.
    # T[(0,0),(0,0),(0,0)] = 1. Rank 1. ✓
    # 1x1x2: T[(0,0),(0,0),(0,0)]=1, T[(0,0),(0,1),(1,0)]=1
    # Can this be rank 1? u⊗v⊗w where u[0]=a, v[0]=b, v[1]=c, w[0]=d, w[1]=e
    # a*b*d=1, a*c*e=1. Set a=1,b=1,d=1,c=1,e=1. Yes, rank 1!
    # But lower bound: max(1*1, 1*2, 2*1) = max(1,2,2) = 2. Contradiction!
    
    # So my lower bound formula is wrong, or my understanding is wrong.
    # Actually, the lower bound rank ≥ max(nm, mp, pn) is NOT a valid lower bound in general.
    # The valid lower bounds are more subtle (border rank, etc.)
    
    # So maybe rank(3x3x7) really is small! Let me just try rank 3 and verify.
    
    # Schoolbook decomposition: rank = n*m*p = 63
    # T[(i,j),(j,k),(k,i)] = 1 for each (i,j,k)
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
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": U, "V": V, "W": W}
    

# EVOLVE-BLOCK-END
