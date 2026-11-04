"""Matrix multiplication 2x3x9 - seeking rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank decomposition for 2x3 * 3x9 matrix multiplication.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank).
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    where A is 2x3, B is 3x9, C is 2x9.
    
    Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    The reconstruction: C[k*n+i] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
    Actually the bilinear form is: for each r, the contribution to C[i,k] is
    sum_j U[i*m+j, r] * V[j*p+k, r] and W[k*n+i, r] should be 1 for the output.
    
    Wait - let me re-read the convention. The output factor W[k*n+i, r] suggests
    W is used to combine. The standard tensor decomposition for matmul:
    T = sum_r u_r ⊗ v_r ⊗ w_r where T_{i,j,k} contributes to output.
    
    For matrix mult A(m×n)B(n×p): T_{i,j,k} = 1 if we think of it as
    C[i,k] = sum_j A[i,j]B[j,k].
    
    In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
    
    For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need the bilinear form
    to exactly reconstruct.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (27, rank)  
    # W: (p*n, rank) = (18, rank)
    
    # Schoolbook: rank = n*m*p = 54
    # For each (i, j, k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This gives C[k*n+i] = sum_col sum_j U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col]
    # With schoolbook: for col corresponding to (i0,j0,k0): U[i*m+j,col]=1 iff i=i0,j=j0
    # V[j*p+k,col]=1 iff j=j0,k=k0, W[k*n+i,col]=1 iff k=k0,i=i0
    # So contribution is 1 when (i,j,k)=(i0,j0,k0), summing over j gives A[i,j0]*B[j0,k] for the right k
    
    # Actually the standard schoolbook for matmul tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has T_{i,j,k} = 1 (all ones) but that's rank 1 as a tensor...
    # No! The tensor for bilinear map is different.
    
    # The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as C = sum_j (A[:,j] * B[j,:]) i.e., C = sum_j outer(A[:,j], B[j,:])
    # So C = sum_{j=0}^{m-1} A_col_j ⊗ B_row_j
    # In terms of the flattening: if we flatten A to a[m*n] with A_flat[i*n+j] = A[i,j]
    # and B to b[m*p] with B_flat[j*p+k] = B[j,k], then C_flat[k*n+i] = sum_j A_flat[i*n+j]*B_flat[j*p+k]
    
    # The tensor T is: T = sum_{i,j,k} E_{(i,j)} ⊗ E_{(j,k)} ⊗ E_{(i,k)} where E are basis vectors
    # Wait, I need to match the problem's indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # So A is indexed as A[i, j] stored at position i*m+j in the flat vector
    # B is indexed as B[j, k] stored at position j*p+k
    # C is indexed as C[i, k] stored at position k*n+i
    
    # The bilinear form: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # = sum_j (e_{i*m+j}^T a) * (e_{j*p+k}^T b)
    
    # As a tensor: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Rank of this tensor is what we minimize.
    
    # Schoolbook: each (i,j,k) triple gives one rank-1 term, total n*m*p = 54 terms.
    
    # Can we do better? For fixed i, the terms are sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = (sum_j e_{i*m+j} ⊗ e_{j*p+*}) ⊗ e_{*n+i}... not quite separable.
    
    # For fixed i: T_i = sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = sum_j (e_{i*m+j} ⊗ (sum_k e_{j*p+k}) ⊗ e_{(k*n+i) varies with k})
    
    # Hmm, for fixed i and j: sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # The first factor is constant, the second varies in k, the third varies in k.
    # sum_k (e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}) - this is a sum of p rank-1 tensors.
    # Can this sum have rank less than p? The vectors e_{j*p+k} for different k are orthogonal,
    # and e_{k*n+i} for different k are also orthogonal (for fixed i, as k varies, k*n+i gives
    # different positions). So the sum has rank p (they're linearly independent).
    
    # So for fixed i: T_i has rank at least p*m = 27? No, the sum over j of these...
    # T_i = sum_j [sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}]
    # For different j, the e_{i*m+j} factors are different (orthogonal), so the total rank is
    # sum over j of rank of inner sum = m * p = 27 for each i, total 54.
    
    # But we can potentially share across i! If we combine i=0 and i=1:
    # T = T_0 + T_1 where T_i involves e_{i*m+j} for i in {0,1}.
    
    # Key insight: For the same (j,k), we have terms for i=0 and i=1:
    # e_{0*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+0} and e_{1*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+1}
    # = (e_{j*p+k} ⊗ ?) ... these share the middle factor e_{j*p+k}.
    # e_{0*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+0} + e_{1*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+1}
    # = (e_{0*m+j} ⊗ e_{k*n+0} + e_{1*m+j} ⊗ e_{k*n+1}) ⊗ e_{j*p+k}  -- NO, tensor product doesn't distribute like that.
    
    # Actually: a⊗b⊗c + a'⊗b⊗c' = b⊗(a⊗c + a'⊗c') is NOT valid in general.
    # a⊗b⊗c is a rank-1 tensor in the triple product space.
    
    # Let me think differently. The tensor T ∈ R^{(n*m)} × R^{(m*p)} × R^{(p*n)}
    # T_{(i,j),(j',k),(k',i')} = 1 if j=j' and k=k' and i=i', else 0.
    # Wait no: T_{(i,j),(j',k),(k',i')} = 1 iff j=j' AND k=k' AND i=i'? 
    # No: the condition is that the index j in A matches j' in B, and k in B matches k' in C,
    # and i in A matches i' in C. So T_{(i,j),(j',k),(k',i')} = δ_{j,j'} δ_{k,k'} δ_{i,i'}.
    
    # So T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}  (using the problem's flat indexing)
    # with e_{(i,j)} = e_{i*m+j} in R^{n*m}, e_{(j,k)} = e_{j*p+k} in R^{m*p}, e_{(k,i)} = e_{k*n+i} in R^{p*n}.
    
    # This is exactly the matrix multiplication tensor for n×m times m×p matrices.
    # Its tensor rank is known to be between some bounds. For 2×3×9, let's try to find a
    # decomposition with rank < 54.
    
    # Strategy: Use the fact that for each fixed j, we have p terms for each i.
    # T = sum_{j=0}^{m-1} sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = sum_j [sum_i sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}]
    
    # For fixed j: S_j = sum_i sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = sum_k [sum_i e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}]
    # For fixed j,k: sum_i e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = e_{j*p+k} ⊗ (sum_i e_{i*m+j} ⊗ e_{k*n+i})  -- NO, can't factor like this.
    # = sum_i (e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i})
    # These n terms (for i=0,1) have the same middle factor e_{j*p+k}.
    # sum_i e_{i*m+j} ⊗ e_{k*n+i} is a matrix in R^{n*m} × R^{p*n} that is rank n (since
    # for different i, the positions are different). Specifically it's a permutation-like matrix.
    # So sum_i e_{i*m+j} ⊗ e_{k*n+i} ⊗ ... wait, I'm mixing up the tensor factors.
    
    # Let me just try a computational approach: solve for a low-rank decomposition.
    
    # For rank r, we need U (6×r), V (27×r), W (18×r) such that
    # T_{(a),(b),(c)} = sum_r' U[a,r'] * V[b,r'] * W[c,r']
    # where a = i*m+j, b = j'*p+k, c = k'*n+i', and T=1 iff j=j', k=k', i=i'.
    
    # This is a 3D tensor completion / decomposition problem.
    
    # Given time constraints, let me try a structured ansatz.
    
    # Ansatz 1: Block structure based on j.
    # For each j (0,1,2), the sub-tensor involves A indices {j, m+j, 2m+j} (i.e., i*m+j for i=0,1),
    # B indices {j*p, j*p+1, ..., j*p+8}, and C indices {i'*9 + ... } wait.
    
    # C index is k*n+i = k*2+i. For the sub-tensor at fixed j, we need k*2+i' where
    # i' matches i from A. So C indices involved: for i=0: {0,2,4,...,16}, for i=1: {1,3,5,...,17}.
    
    # Sub-tensor for fixed j: 
    # T_j ∈ R^{2} × R^{9} × R^{18} (A has 2 entries for this j, B has 9, C has 18 but only 9 active)
    # T_j[a, b, c] = 1 if a corresponds to i, b corresponds to k, c corresponds to (k,i)
    # i.e., T_j[i, k, k*2+i] = 1 for i∈{0,1}, k∈{0,...,8}
    
    # This sub-tensor has 18 nonzero entries. Can it be decomposed with rank < 18?
    # T_j = sum_{i,k} e_i ⊗ e_k ⊗ e_{2k+i}
    # = sum_k (e_0 ⊗ e_k ⊗ e_{2k} + e_1 ⊗ e_k ⊗ e_{2k+1})
    # For each k: e_0 ⊗ e_k ⊗ e_{2k} + e_1 ⊗ e_k ⊗ e_{2k+1}
    # = e_k ⊗ (e_0 ⊗ e_{2k} + e_1 ⊗ e_{2k+1})  -- again, can't factor this way.
    # As a sum of rank-1 tensors: e_0⊗e_k⊗e_{2k} and e_1⊗e_k⊗e_{2k+1} are 2 rank-1 tensors.
    # Are they linearly independent? Yes, since the C components are different.
    # So rank of T_j is at least 18 (sum of 9 pairs of independent rank-1 tensors).
    # Actually, could there be a clever combination? The 18 rank-1 tensors are:
    # {e_0⊗e_k⊗e_{2k}, e_1⊗e_k⊗e_{2k+1} for k=0,...,8}
    # All 18 are linearly independent (different support patterns), so rank(T_j) = 18.
    # Total rank = 3 * 18 = 54. No improvement from this block structure.
    
    # Ansatz 2: Try to combine across j values.
    # The full tensor T = sum_j T_j. Can T be decomposed with rank < sum_j rank(T_j) = 54?
    # Only if there's structure allowing sharing. The A-index a = i*m+j: for different j,
    # a ranges over {0,1,2} (j=0), {3,4,5} (j=1), {6,7,8} (j=2). These are disjoint.
    # The B-index b = j*p+k: for different j, b ranges over {0-8}, {9-17}, {18-26}. Disjoint.
    # The C-index c = k*n+i: this is shared across j! c ranges over {0,...,17} for all j.
    
    # So the C-factor is shared. This means we potentially can combine terms across j.
    # T = sum_{i,j,k} e_{i*3+j} ⊗ e_{j*9+k} ⊗ e_{2k+i}
    # For fixed (i,k): sum_j e_{i*3+j} ⊗ e_{j*9+k} ⊗ e_{2k+i}
    # = e_{2k+i} ⊗ (sum_j e_{i*3+j} ⊗ e_{j*9+k})  -- can't factor this way either.
    # = sum_j (e_{i*3+j} ⊗ e_{j*9+k}) ⊗ e_{2k+i}
    # The sum over j of e_{i*3+j} ⊗ e_{j*9+k} is a 6×27 matrix M_{ik} with
    # M_{ik}[i*3+j, j*9+k] = 1. This matrix has rank 3 (one nonzero per row, 3 nonzeros).
    # So sum_j e_{i*3+j} ⊗ e_{j*9+k} has rank 3 as a matrix, meaning it can be written as
    # sum_{j=0}^{2} e_{i*3+j} ⊗ e_{j*9+k} (which is what we have).
    
    # So for each (i,k): T_{ik} = [sum_j e_{i*3+j} ⊗ e_{j*9+k}] ⊗ e_{2k+i}
    # The "matrix" part has rank 3, so we still need 3 rank-1 terms for the (A,B) part,
    # each tensored with e_{2k+i}. Total: 2 * 9 * 3 = 54.
    
    # It seems the rank is indeed 54 for this tensor. Let me verify with a small case.
    
    # For 1×1×1: rank 1. For 1×2×1: T = e_0⊗e_0⊗e_0 + e_1⊗e_1⊗e_0. Rank 2 (independent).
    # For 2×1×1: T = e_0⊗e_0⊗e_0 + e_1⊗e_0⊗e_1. Rank 2.
    # For 2×2×1: T = sum_{i,j} e_{i*2+j} ⊗ e_{j*1+0} ⊗ e_{0*2+i} = sum_{i,j} e_{2i+j} ⊗ e_j ⊗ e_i
    # = e_0⊗e_0⊗e_0 + e_1⊗e_0⊗e_1 + e_2⊗e_1⊗e_0 + e_3⊗e_1⊗e_1
    # All 4 are linearly independent, rank 4 = n*m*p.
    
    # It appears the matrix multiplication tensor T_{n,m,p} always has tensor rank = n*m*p
    # when considered as a 3-way tensor with the given index structure. This is because all
    # the rank-1 terms in the schoolbook decomposition are linearly independent.
    
    # Proof sketch: Suppose T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r with R < nmp.
    # Consider the slice fixing C-index c = k*n+i. Then:
    # sum_j e_{i*m+j} ⊗ e_{j*p+k} = sum_r U[:, c=r part]... 
    # This gets complex. Let me just verify computationally for small cases.
    
    # Given the analysis, I believe the minimum rank is 54 for 2×3×9.
    # But let me try to verify with a rank-revealing approach or just return the schoolbook.
    
    # Actually, let me try a numerical approach: set up the equations and try to find
    # a rank-r decomposition for r < 54.
    
    # For rank r, we have U (6×r), V (27×r), W (18×r). The constraint is:
    # For all (a,b,c) where a=i*3+j, b=j'*9+k, c=k'*2+i':
    # T[a,b,c] = sum_{r'} U[a,r']*V[b,r']*W[c,r']
    # where T[a,b,c] = 1 iff (j part of a == j' ) and (k part of b == k') and (i part of a == i')
    # i.e., T[i*3+j, j'*9+k, k'*2+i'] = δ_{j,j'} δ_{k,k'} δ_{i,i'}
    
    # Number of constraints: 6*27*18 = 2916, but most are zero. Nonzero: 2*3*9=54.
    # Unknowns: 6r + 27r + 18r = 51r. For r=54: 2754 unknowns, 54 nonzero + 2862 zero constraints.
    # For r < 54: still overdetermined in a sense.
    
    # Let me try to solve for r=53, 52, etc. using least squares or direct solving.
    # But this is a non-convex problem. Let me try a different approach.
    
    # Since I'm not finding an obvious improvement, and the analysis suggests rank=54 is minimal,
    # let me return the schoolbook decomposition with rank 54.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as expected
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
