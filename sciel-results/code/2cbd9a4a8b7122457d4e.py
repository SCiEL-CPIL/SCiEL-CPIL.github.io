import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition (Rank 48) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy: Search for lower rank decompositions using structured patterns
    # For matrix multiplication tensors, optimal decompositions often have
    # specific sparsity patterns. We'll search over support patterns.
    
    # The tensor has n*m*p = 48 non-zero entries.
    # Each rank-1 component u ⊗ v ⊗ w can cover multiple non-zero entries
    # if the supports align properly.
    
    # Key insight: For a rank-1 component to be consistent with T,
    # if u[a]≠0, v[b]≠0, w[c]≠0, then T[a,b,c] must be non-zero (or the
    # product must cancel with other components).
    
    # Let's try a different approach: use the fact that we can pad to 4x4x4
    # and use known results, then restrict.
    
    # Actually, let's try a more direct approach:
    # Search for decompositions where each factor column has limited support.
    
    # For rank R, we need R vectors u_r, v_r, w_r such that
    # sum_r u_r ⊗ v_r ⊗ w_r = T
    
    # This is equivalent to: for each (a,b,c) with T[a,b,c]=1,
    # sum_r u_r[a] * v_r[b] * w_r[c] = 1
    # and for each (a,b,c) with T[a,b,c]=0,
    # sum_r u_r[a] * v_r[b] * w_r[c] = 0
    
    # Let's enumerate the non-zero pattern of T
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    # Number of non-zero entries
    nnz = len(nz_indices)  # 48
    
    # For each rank R, we have R*3 variables (u, v, w entries)
    # and constraints from all (a,b,c) positions.
    # This is a system of polynomial equations.
    
    # Alternative: Use the structure more directly.
    # The matrix multiplication tensor M_{n,m,p} has a known lower bound
    # and for small dimensions, optimal decompositions are known.
    
    # For M_{3,4,4}:
    # Lower bound: max(12, 16, 12) = 16
    # Upper bound: 48 (schoolbook)
    # Known results suggest rank is between 20 and 30 or so.
    
    # Let's try a structured search:
    # Assume each column of U, V, W has at most s non-zero entries.
    # For small s, we can enumerate the support patterns.
    
    # But this is still combinatorially expensive.
    
    # Better approach: Use the fact that the tensor is "structured" and
    # apply a variant of the "circuit complexity" approach.
    
    # Actually, let's try a pragmatic approach:
    # 1. Start with the schoolbook rank-48 decomposition
    # 2. Try to merge pairs of rank-1 components into a single one
    #    (if they have compatible support patterns)
    # 3. If merging fails, try other local transformations
    
    # For merging: Two rank-1 components u1⊗v1⊗w1 and u2⊗v2⊗w2
    # can potentially be replaced by a single component if they have
    # the same support pattern (up to scaling).
    
    # Let's check if any pairs in the schoolbook decomposition have
    # compatible structures.
    
    # In the schoolbook decomposition, each component is a "basis" component:
    # Component (i,j,k): u has 1 at position i*m+j, v has 1 at j*p+k, w has 1 at k*n+i
    # These are all orthogonal, so they can't be merged directly.
    
    # However, we can try to find a different basis.
    
    # Let's try a completely different approach: use the known fact that
    # M_{4,4,4} has rank at most 49 (or some known bound), and see if we
    # can restrict to the 3x4x4 sub-tensor.
    
    # Actually, for this problem, let's try a more aggressive search:
    # Use random restarts with a focus on sparse solutions.
    
    # The key insight from tensor decomposition theory: for structured tensors
    # like matrix multiplication, the optimal decomposition often has a
    # "block" structure where groups of indices are treated together.
    
    # Let's try to find a decomposition with rank R where each factor
    # column has a specific sparsity pattern.
    
    # For simplicity, let's try ranks from 25 to 40 and use a
    # gradient-based approach with sparsity regularization.
    
    # Actually, given the time constraints, let's try a more direct approach:
    # Use the fact that we can represent the multiplication as a sequence
    # of smaller multiplications.
    
    # For 3x4 * 4x4:
    # We can split the 3x4 matrix into rows and use Strassen-like tricks.
    
    # Let's try to construct a decomposition explicitly.
    
    # Approach: Use the fact that M_{3,4,4} can be embedded in M_{4,4,4}.
    # If we have a rank-R decomposition of M_{4,4,4}, we can restrict it
    # to the 3x4x4 sub-tensor by zeroing out the 4th row/column.
    
    # But we don't have a known low-rank decomposition of M_{4,4,4} either.
    
    # Let's try a different structural approach:
    # Use the "Lanczos" or "Wiedemann" type algorithm for tensor rank.
    
    # Actually, let's just do a more careful ALS with better initialization
    # and more iterations, focusing on the most promising ranks.
    
    # The key improvement: use the structure of the tensor to initialize
    # the factors in a way that respects the sparsity pattern.
    
    # For each rank R, initialize U, V, W such that each column has
    # support aligned with the non-zero pattern of T.
    
    # Let's try ranks that are multiples of small numbers, as they
    # may correspond to block structures.
    
    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 40, 42, 44, 46]
    ranks_to_try = [r for r in ranks_to_try if r < best_rank]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try multiple initializations
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + R * 100 + trial)
            
            # Initialize with a bias toward the non-zero pattern
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Optimize U: for each column r, solve for U[:,r]
                # given V and W
                B_u = np.zeros((R, m * p * p * n))
                for r in range(R):
                    B_u[r] = np.outer(V[:, r], W[:, r]).reshape(-1)
                
                try:
                    U, _, _, _ = np.linalg.lstsq(B_u.T, T.reshape(n * m, -1), rcond=None)
                except:
                    pass
                
                # Optimize V
                B_v = np.zeros((R, n * m * p * n))
                for r in range(R):
                    B_v[r] = np.outer(U[:, r], W[:, r]).reshape(-1)
                
                try:
                    V, _, _, _ = np.linalg.lstsq(B_v.T, T.reshape(m * p, -1), rcond=None)
                except:
                    pass
                
                # Optimize W
                B_w = np.zeros((R, n * m * m * p))
                for r in range(R):
                    B_w[r] = np.outer(U[:, r], V[:, r]).reshape(-1)
                
                try:
                    W, _, _, _ = np.linalg.lstsq(B_w.T, T.reshape(p * n, -1), rcond=None)
                except:
                    pass
                
                # Compute error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                error = np.linalg.norm(T - recon)
                
                if error < 1e-10:
                    break
            
            # Check if we found an exact decomposition
            recon = np.zeros_like(T)
            for r in range(R):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.linalg.norm(T - recon)
            
            if error < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                recon_rounded = np.zeros_like(T)
                for r in range(R):
                    recon_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                error_rounded = np.linalg.norm(T - recon_rounded)
                
                if error_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    found = True
                    break
                else:
                    # Use the unrounded version if it's exact
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
