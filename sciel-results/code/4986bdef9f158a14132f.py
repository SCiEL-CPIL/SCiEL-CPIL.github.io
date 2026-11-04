"""Matrix multiplication 3,3,3 - rank-minimal decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
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
    
    # Build the multiplication tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple
    # Actually, the tensor for matrix mult C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this context)
    # Wait no. The tensor M[i,j,k] where the decomposition is:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # For matrix multiplication, M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # Actually M is the identity in some sense... let me think more carefully.
    
    # The tensor for matrix multiplication of size n x p = n x m * m x p is:
    # T[i,j,k] = delta[i,j,k] where the product C[i,k] = sum_j A[i,j] B[j,k]
    # The rank-1 decomposition terms are: U[i,j] * V[j,k] * W[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the all-ones tensor of size n x m x p
    
    # Verify a decomposition
    def verify(U, V, W):
        """Check if U[i,j,r]*V[j,k,r]*W[i,k,r] sums to 1 for all i,j,k."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is expensive for large rank, so do it smartly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U_arr[idx_u, :], V_arr[idx_v, :] * W_arr[idx_w, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-20 decomposition using algebraic methods
    # The known result: 3x3 matrix multiplication has tensor rank 20 (Haviv-Shpilka 2018)
    # But the explicit construction is complex. Let's try a structured search.
    
    # Approach: Use the fact that the all-ones 3x3x3 tensor can be decomposed
    # Try random rank-20 decompositions with structured patterns
    
    # First, let's try a systematic approach using circulant/group structure
    # Map indices to Z_5 x Z_5 and use characters
    
    # Actually, let me try a different approach: start with schoolbook (rank 27)
    # and try to find linear dependencies among the 27 rank-1 tensors to reduce rank.
    
    # The tensor space: each rank-1 term is a matrix in the space of (9x9) matrices
    # (since we're looking at U[9], V[9], W[9])
    # A rank-1 term r gives a 9x9 matrix: for row (i,j) and col (i',k'), 
    # the entry is 1 if i=i' and j=k'... no that's not right.
    
    # Let me think about this differently. The multiplication tensor T in R^{9x9x9}
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k'*n+i' where i=i' (matching first index)
    # Wait, let me re-read the convention.
    
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i  (note: k*n+i, not i*n+k)
    
    # The decomposition: T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # And for matrix multiplication, this sum should be 1 for all valid (i,j,k) triples
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # But what about entries where the indices don't form a valid triple?
    # E.g., if a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where j1 != j2 or k2 != k3 or i1 != i3
    # Then T[a,b,c] should be 0.
    
    # So the full tensor T is defined on all of R^{9x9x9}, with T[a,b,c] = 1 if the 
    # (i,j,k) extracted from a,b,c are consistent, else 0.
    
    # Let me build this tensor explicitly
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Now, each rank-1 decomposition term corresponds to a vector u in R^{n*m},
    # v in R^{m*p}, w in R^{p*n}, and contributes outer(u,v,w) to T.
    # We need sum_r outer(u_r, v_r, w_r) = T
    
    # For rank minimization, we need to find the minimum rank such that T can be
    # written as such a sum.
    
    # Strategy: Try random rank-20 decompositions and check if they work.
    # This is unlikely to succeed with random vectors, but let's try structured ones.
    
    # Better strategy: Use the known algebraic structure.
    # The 3x3 matrix multiplication tensor is equivalent to the 3x3x3 all-ones tensor
    # with consistency constraints.
    
    # Let me try: express T as a sum of rank-1 tensors where the vectors have
    # a special form related to roots of unity.
    
    # Use omega = exp(2*pi*i/5) and construct decompositions based on Z_5 characters.
    # The idea: embed the 3x3 indices into 5x5 and use the 5x5 DFT.
    
    # Actually, let me try a more direct numerical approach:
    # 1. Start with rank 27 (schoolbook)
    # 2. Try to find a rank-20 sub-decomposition by solving a linear system
    
    # The key insight: T can be written as sum of 27 rank-1 terms (schoolbook).
    # We want to find if there's a linear combination of fewer terms that gives the same T.
    
    # Represent each rank-1 term as a vector in R^{81} (flattening the 9x9 matrix for fixed "middle" index)
    # Actually, let's flatten T into R^{729} (9*9*9)
    
    T_flat = T.flatten()
    
    # Each rank-1 term u,v,w gives a 729-dim vector: outer(u,v,w).flatten()
    # Schoolbook gives 27 such vectors. Check if they're linearly independent.
    
    # Build the 27 schoolbook vectors
    svectors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                svectors.append(np.outer(np.outer(u, v), w).flatten())
    
    S = np.array(svectors).T  # 729 x 27 matrix
    
    # Check rank of S
    rank_S = np.linalg.matrix_rank(S)
    # If rank_S < 27, there are linear dependencies
    
    # For a rank-r decomposition, we need r vectors whose span contains T_flat
    # and which can be written as outer products.
    
    # This is complex. Let me try a different approach:
    # Randomized search for rank-20 decompositions with specific structure.
    
    # Use the fact that for the all-ones tensor (without consistency constraints),
    # the rank is 9 (using rank-1 terms from the 3x3 identity).
    # But with consistency constraints, it's harder.
    
    # Let me try: parametrize the decomposition using 20 terms where each term
    # is indexed by an element of some structure.
    
    # Alternative: Use numpy to try to find a low-rank decomposition numerically.
    # Reshape T as a 3-way tensor and try tensor rank minimization.
    
    # This is computationally hard in general, but for 3x3x3 it might be tractable.
    
    # Let me try a simple randomized approach:
    # Generate random rank-20 decompositions and check if they match T.
    
    # But random won't work well. Let's use a structured approach.
    
    # Key idea: The 3x3 matrix multiplication can be related to the 5x5 case
    # where the rank is 5^2 = 25 (schoolbook) or potentially lower.
    
    # Actually, I recall that the best known upper bound for 3x3 matrix mult
    # is 20, achieved by Haviv and Shpilka. The construction uses the 
    # "laser method" with Z_5 x Z_5.
    
    # Let me try to implement a version of this.
    # The construction maps 3x3 indices to subsets of {0,1,2,3,4} and uses
    # the 5x5 DFT to generate rank-1 terms.
    
    # Time check
    if time.time() - start_time > deadline:
        return schoolbook()
    
    # Let me try a numerical optimization approach:
    # Minimize ||sum_r outer(u_r,v_r,w_r) - T||^2 subject to sum_r ... = T
    # Start from schoolbook and try to zero out some coefficients.
    
    # Actually, let me try the most direct approach: 
    # Search over rank-20 decompositions where the vectors take values in 
    # {0, 1, -1} or {0, 1/2, -1/2, 1, -1} (half-integers).
    
    # This is a huge search space, but with structure it might work.
    
    # For now, let's just return the schoolbook decomposition since finding
    # an explicit rank-20 decomposition within the time limit is challenging.
    
    # But let me try one more thing: check if the 27 schoolbook vectors
    # have a null space, and if so, use that to reduce the rank.
    
    # Compute SVD of S to find the rank and null space
    U_svd, sigma_svd, Vt_svd = np.linalg.svd(S, full_matrices=False)
    
    # The effective rank is the number of significant singular values
    effective_rank = np.sum(sigma_svd > 1e-10)
    
    # If effective_rank < 27, we might be able to reduce the decomposition rank
    # But the reduced terms must still be valid rank-1 tensors (outer products)
    
    # This is the key constraint that makes it hard.
    
    # Let me try a final approach: use scipy's tensor decomposition or similar.
    
    # Actually, let me just try random rank-20 attempts with a timeout.
    
    np.random.seed(42)
    
    def try_random_rank20(attempts=1000):
        """Try random rank-20 decompositions."""
        for _ in range(attempts):
            if time.time() - start_time > deadline:
                return None
            
            # Generate random rank-20 decomposition
            U = np.random.randn(n*m, 20)
            V = np.random.randn(m*p, 20)
            W = np.random.randn(p*n, 20)
            
            # Check if sum_r outer(U[:,r], V[:,r], W[:,r]) approximates T
            # Efficient check: for each (a,b,c) where T[a,b,c]=1, verify
            # sum_r U[a,r]*V[b,r]*W[c,r] ≈ 1
            # And for (a,b,c) where T[a,b,c]=0, verify ≈ 0
            
            # This is expensive, so just check a few entries
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        s = np.dot(U[a,:], V[b,:]*W[c,:])
                        error += (s - 1.0)**2
            
            # Also check some zero entries
            for _ in range(10):
                a = np.random.randint(n*m)
                b = np.random.randint(m*p)
                c = np.random.randint(p*n)
                # Check if this is a zero entry
                # Extract indices
                i1, j1 = divmod(a, m)
                j2, k2 = divmod(b, p)
                k3, i3 = divmod(c, n)
                if j1 == j2 and k2 == k3 and i1 == i3:
                    continue  # This is a one entry
                s = np.dot(U[a,:], V[b,:]*W[c,:])
                error += s**2
            
            if error < 1e-6:
                # Convert to lists with half-integer rounding
                U_list = [[round(U[i][j] * 2) / 2 for j in range(20)] for i in range(n*m)]
                V_list = [[round(V[i][j] * 2) / 2 for j in range(20)] for i in range(m*p)]
                W_list = [[round(W[i][j] * 2) / 2 for j in range(20)] for i in range(p*n)]
                
                # Verify the rounded version
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try random search
    result = try_random_rank20(500)
    if result is not None:
        return result
    
    # Return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
