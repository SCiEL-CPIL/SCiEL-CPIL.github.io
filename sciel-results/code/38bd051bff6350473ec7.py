"""
Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

We use the "sub-tensor peeling" approach:
1. Represent the 5x5x5 matrix multiplication tensor T of shape (25, 25, 25).
   T[i,j,k] = 1 if (i,j) is a position in A, (j,k) in B, and the product contributes to C[i,k].
   Specifically, for standard matrix multiplication C = A @ B:
   C[i,k] = sum_j A[i,j] * B[j,k]
   So the tensor has T[i,j,k] = 1 for all i,j,k (since each triple contributes exactly once to the product term).
   
   Wait, that's not right. The tensor for matrix multiplication is:
   T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
   Actually, the standard tensor is defined as:
   T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
   where the multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
   
   So the tensor is actually the all-ones tensor of shape (n, m, p)? No.
   
   Let me reconsider. The matrix multiplication tensor T of format (n,m,p) is defined by:
   T[i, j, k] = 1 if the (i,k)-entry of C uses A[i,j] and B[j,k].
   Since C[i,k] = sum_j A[i,j] B[j,k], each term A[i,j]B[j,k] contributes to C[i,k].
   So T[i, j, k] = 1 for all valid i, j, k. That would make it the all-ones tensor.
   
   But that's not a useful tensor decomposition problem. Let me re-read the task.
   
   The task says: "Find a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix by a 5 by 5 matrix."
   
   The standard matrix multiplication tensor is indeed T[i,j,k] = 1 for all i,j,k in the appropriate ranges, but the tensor is viewed as a 3-tensor where:
   - First index i ranges over rows of A (and C)
   - Second index j ranges over columns of A / rows of B
   - Third index k ranges over columns of B (and C)
   
   And the tensor is: T[i,j,k] = 1 if we multiply A[i,j] by B[j,k] to get a contribution to C[i,k].
   
   So yes, T[i,j,k] = 1 for all i,j,k. The tensor is the all-ones tensor of shape (5,5,5).
   
   But wait, that can't be right either, because the all-ones tensor of shape (n,m,p) has rank 1 (it's just the outer product of three all-ones vectors).
   
   I think I'm confusing myself. Let me look at this differently.
   
   The matrix multiplication tensor is a 3-tensor T in U ⊗ V ⊗ W where:
   - U has dimension n*p (for the output C entries)
   - V has dimension n*m (for the input A entries)  
   - W has dimension m*p (for the input B entries)
   
   No, that's not standard either.
   
   Let me use the standard definition. The matrix multiplication tensor T_{n,m,p} is a tensor in R^{n} ⊗ R^{m} ⊗ R^{p} defined by:
   T_{i,j,k} = 1 if and only if... 
   
   Actually, I recall now. The standard way is:
   The tensor T is in R^{nm} ⊗ R^{mp} ⊗ R^{pn} or similar, depending on the convention.
   
   From the task: "The output factor follows the official trace convention W[k*n+i,r]."
   
   So W has shape (p*n, rank) and the entry W[k*n + i, r] corresponds to the contribution to C[i,k] from the r-th rank-1 component.
   
   U has shape (n*m, rank): U[i*m + j, r] corresponds to A[i,j]
   V has shape (m*p, rank): V[j*p + k, r] corresponds to B[j,k]
   W has shape (p*n, rank): W[k*n + i, r] corresponds to C[i,k]
   
   The decomposition says:
   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
   
   where T[i,j,k] = 1 for all i,j,k (since each product term A[i,j]*B[j,k] contributes to C[i,k]).
   
   Wait, that still gives the all-ones tensor. But the rank of the all-ones tensor of shape (n,m,p) is 1.
   
   I must be misunderstanding the tensor structure. Let me think again.
   
   Oh! I see the issue. The tensor is NOT the all-ones tensor. The tensor T[i,j,k] represents whether the (i,j) entry of A and the (j,k) entry of B contribute to the (i,k) entry of C. But in matrix multiplication, for a FIXED i and k, we sum over ALL j. So T[i,j,k] = 1 for all i,j,k.
   
   But that gives rank 1, which is trivially wrong for matrix multiplication.
   
   I think the key insight is that the tensor is not T[i,j,k] but rather a tensor that encodes the LINEAR MAP from (A,B) to C. The tensor is in (R^{nm})^* ⊗ (R^{mp})^* ⊗ R^{pn} or something like that.
   
   Let me use the standard formulation. The matrix multiplication tensor is a 3-tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (flattened) such that:
   <T, (A, B, C)> = trace(C^T (A B))
   
   Or equivalently, T is defined by:
   T[i*m+j, j'*p+k, i'*n+k'] = delta_{i,i'} * delta_{k,k'} * delta_{j,j'}
   
   Wait, let's be more careful. Using the trace convention:
   C = A B, so C[i,k] = sum_j A[i,j] B[j,k]
   
   The tensor T satisfies:
   sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * X[i,k] for some X
   
   Actually, the standard bilinear form is:
   <A, B, C> = sum_{i,j,k} A[i,j] B[j,k] C^T[i,k] = trace(C^T A B)
   
   So T[i*m+j, j*p+k, i*n+k] = 1 (using the flattening conventions from the task).
   
   Let me define:
   - Index for A: a = i*m + j, where i ∈ [0,n), j ∈ [0,m)
   - Index for B: b = j*p + k, where j ∈ [0,m), k ∈ [0,p)
   - Index for C: c = k*n + i, where k ∈ [0,p), i ∈ [0,n)  [trace convention: W[k*n+i,r]]
   
   Then T[a, b, c] = 1 if and only if:
   a = i*m + j, b = j'*p + k, c = k'*n + i'
   and i = i', j = j', k = k'
   
   So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
   
   This is a sparse tensor! It has exactly n*m*p = 125 non-zero entries.
   
   The rank of this tensor is what we want to minimize. The schoolbook algorithm uses rank = n*m*p = 125, with each rank-1 term being:
   e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
   
   We need to find a decomposition of lower rank.
   
   For 5x5x5, the best known rank is 49 (from various sources, the lower bound is around 47-49).
   
   Let me implement a randomized algorithm to find a low-rank decomposition.
   
   Approach:
   1. Start with the tensor T (sparse, 125 non-zeros).
   2. Use randomized linear algebra to find rank-1 components.
   3. Subtract them and iterate.
   4. For the residual, use exact integer arithmetic.
   
   Actually, for a practical approach within 120 seconds, let me try a different strategy:
   - Use the fact that for small sizes, we can try to find a decomposition via integer linear programming or greedy methods.
   - Or, use known constructions for 5x5x5.
   
   Let me try a greedy approach:
   1. Represent the tensor as a 3D array of size (25, 25, 25) with 125 ones.
   2. Iteratively find a rank-1 tensor that has maximum inner product with the residual.
   3. Subtract it.
   4. Repeat until the residual is zero.
   
   But this greedy approach may not give optimal rank.
   
   Alternative: Use the "peeling" idea. Find a set of rank-1 tensors that are linearly independent and sum to a large portion of T, subtract, and recurse.
   
   Given time constraints, let me try a practical approach:
   - Use a randomized algorithm over a large prime field to find a decomposition.
   - Then convert to rational/integer form.
   
   Actually, let me try a simpler approach: use the known Strassen-like constructions or just implement a decent greedy algorithm.
   
   For 5x5x5, let me try to find a decomposition of rank ~49-55.
   
   Let me implement a randomized peeling algorithm:
   1. Work over a large prime field (e.g., 10^9+7).
   2. Represent the tensor T.
   3. Randomly sample rank-1 tensors, check if they are "aligned" with T.
   4. Build a basis of such rank-1 tensors.
   5. Subtract and recurse.
   
   This is complex. Let me instead try a direct approach:
   - Use numpy to solve for a decomposition.
   - Start with a random initialization and use iterative refinement.
   
   Given the 120-second limit, let me try a pragmatic approach:
   - Use a known good rank for 5x5x5 (around 49-55).
   - Try to find a decomposition of that rank using randomized methods.
   
   Actually, let me just implement a solid greedy algorithm that should get close to optimal.
"""

import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor T: T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n) = (25, 25, 25)
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Greedy peeling: find rank-1 tensors that maximize inner product with residual
    # A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors
    # Inner product: sum_{a,b,c} T[a,b,c] * u[a] * v[b] * w[c]
    
    # We'll work with rational numbers for exactness
    # But for speed, use floats for search, then convert
    
    residual = T.copy().astype(np.float64)
    components = []  # list of (u, v, w) tuples
    
    # Use randomized initialization for rank-1 components
    np.random.seed(42)
    random.seed(42)
    
    max_iter = 200
    for it in range(max_iter):
        if np.allclose(residual, 0, atol=1e-10):
            break
        
        # Try many random rank-1 tensors, pick the best
        best_score = -np.inf
        best_uvw = None
        
        for trial in range(50):
            # Random rank-1 tensor with entries in {-1, 0, 1} or small integers
            u = np.random.choice([-1, 0, 1], size=n*m)
            v = np.random.choice([-1, 0, 1], size=m*p)
            w = np.random.choice([-1, 0, 1], size=p*n)
            
            # Compute inner product with residual
            # score = sum_{a,b,c} residual[a,b,c] * u[a] * v[b] * w[c]
            # Efficient computation:
            # temp[a,b] = sum_c residual[a,b,c] * w[c]
            # score = sum_{a,b} temp[a,b] * u[a] * v[b]
            temp = np.tensordot(residual, w, axes=([2], [0]))  # shape (n*m, m*p)
            score = np.sum(temp * np.outer(u, v))
            
            if score > best_score:
                best_score = score
                best_uvw = (u.copy(), v.copy(), w.copy())
        
        if best_score < 1e-10:
            # No good component found, break
            break
        
        u, v, w = best_uvw
        components.append((u, v, w))
        
        # Subtract the rank-1 tensor
        rank1 = np.einsum('a,b,c->abc', u, v, w)
        residual -= rank1
    
    # Check if we've decomposed exactly
    if not np.allclose(residual, 0, atol=1e-10):
        # Fallback to schoolbook
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
    
    # Convert to integer/half-integer entries
    rank = len(components)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(components):
        for a in range(n * m):
            U[a][r] = int(u[a])
        for b in range(m * p):
            V[b][r] = int(v[b])
        for c in range(p * n):
            W[c][r] = int(w[c])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
