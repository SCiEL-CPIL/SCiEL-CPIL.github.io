import numpy as np
from numpy.linalg import norm
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Target tensor: T[i*m+j, j'*p+k, k'*n+i'] 
    # should be 1 if j==j', i==i', k==k', else 0
    # This represents C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's use a randomized approach with iterative refinement
    # Start with a high rank and try to reduce it
    
    # First, let's construct the target tensor
    # The tensor T has shape (n*m, m*p, p*n) in the flattened indices
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #              0 otherwise
    
    # Actually, for matrix multiplication, the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already a sum of n*m*p terms, giving rank n*m*p = 105
    
    # But we can do better using the structure of matrix multiplication.
    # Let's try a different approach: exploit low-rank structure.
    
    # For matrix multiplication, we can think of it as:
    # C = A @ B
    # This has a natural bilinear form.
    
    # Let me try a greedy/iterative approach:
    # 1. Start with the naive rank-105 decomposition
    # 2. Try to merge terms or find lower-rank representations
    
    # Alternative: Use the fact that for small matrices, we can enumerate
    # and find exact decompositions using linear algebra.
    
    # Let's try a completely different approach: use the structure of
    # rectangular matrix multiplication.
    
    # For A (n×m) and B (m×p), the product C (n×p) satisfies:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The standard tensor decomposition has rank n*m*p.
    # But we can potentially do better.
    
    # Let's use a randomized tensor rank minimization approach.
    
    # Represent the multiplication as a 3-tensor.
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)}
    # T[a, b, c] where:
    #   a indexes A (row-major: a = i*m + j)
    #   b indexes B (row-major: b = j*p + k)  
    #   c indexes C (with convention W[k*n + i], so c = k*n + i)
    
    # The condition is: T[a,b,c] = 1 iff the indices are "compatible"
    # i.e., j from a equals j from b, i from a equals i from c, k from b equals k from c
    
    # Let's build this tensor and try to find its rank.
    
    def build_target_tensor(n, m, p):
        """Build the target tensor for n×m by m×p matrix multiplication."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j  # index in A
                    b = j*p + k  # index in B
                    c = k*n + i  # index in C (using W[k*n+i] convention)
                    T[a, b, c] = 1.0
        return T
    
    def reconstruct_tensor(U, V, W):
        """Reconstruct tensor from factors: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
        # We want T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is the tensor product
        R = U.shape[1]
        T = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
        for r in range(R):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T
    
    # Let's try a randomized approach:
    # Start with rank R, generate random factors, and try to fit the target tensor.
    # Use iterative methods to converge to an exact solution if it exists at that rank.
    
    # For matrix multiplication, we know:
    # - The naive rank is n*m*p = 105
    # - There are lower bounds (e.g., at least max(n,m,p)*something)
    # - For small cases, we might find better decompositions
    
    # Let's try ranks from low to high and see if we can find an exact decomposition.
    
    # Actually, let's use a smarter approach. For rectangular matrix multiplication,
    # we can use block decomposition or exploit symmetries.
    
    # Let's try a direct approach: since the dimensions are small, we can use
    # linear programming or exhaustive search to find the minimum rank.
    
    # Alternative: use the fact that matrix multiplication rank for n×m×p
    # has known values for small cases. Let me check if 3×5×7 has a known rank.
    
    # For now, let's implement a randomized tensor rank minimization.
    
    def try_rank(n, m, p, rank, max_iter=1000, seed=42):
        """Try to find an exact decomposition of the given rank."""
        np.random.seed(seed)
        
        # Build target tensor
        T_target = build_target_tensor(n, m, p)
        
        # Initialize factors randomly
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Iteratively refine using ALS (Alternating Least Squares)
        # with a sparsity/low-rank constraint
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            # We want to minimize ||T_target - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||
            # This is equivalent to: for each a, minimize ||T_target[a,:,:] - sum_r U[a,r] * (V[:,r]⊗W[:,r])||
            # Flatten: T_target[a] is a (m*p * p*n) vector
            # The design matrix for a is: sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So we want to solve for U[a,:] such that X_a @ U[a,:] ≈ T_target[a].flatten()
            # where X_a[:, r] = V[:,r] ⊗ W[:,r]
            
            X = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                X[:, r] = np.outer(V[:, r], W[:, r])
            
            # For each a, solve the least squares
            for a in range(n*m):
                target_a = T_target[a].flatten()
                # Solve X @ u_a = target_a
                U[a, :] = np.linalg.lstsq(X, target_a, rcond=None)[0]
            
            # Similarly update V and W
            X = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                X[:, r] = np.outer(U[:, r], W[:, r])
            for b in range(m*p):
                # T_target[:,b,:] needs to be reshaped
                target_b = T_target[:, b, :].reshape(-1)  # shape (n*m * p*n)
                V[b, :] = np.linalg.lstsq(X, target_b, rcond=None)[0]
            
            X = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                X[:, r] = np.outer(U[:, r], V[:, r])
            for c in range(p*n):
                target_c = T_target[:, :, c].reshape(-1)  # shape (n*m * m*p)
                W[c, :] = np.linalg.lstsq(X, target_c, rcond=None)[0]
            
            # Check convergence
            T_recon = reconstruct_tensor(U, V, W)
            error = norm(T_target - T_recon, 'fro')
            if error < 1e-10:
                # Found an exact solution
                return U, V, W
        
        return None
    
    # Let's try different ranks
    # Start with the naive rank and see if we can reduce
    best_result = None
    
    # Try ranks from a reasonable lower bound up to the naive rank
    # Lower bound: at least max(n, m, p) for non-trivial cases
    min_rank = max(n, m, p)  # 7
    
    for rank in range(min_rank, min(min_rank + 50, n*m*p + 1)):
        if time.time() > deadline - 5:
            break
            
        result = try_rank(n, m, p, rank, max_iter=200, seed=rank)
        if result is not None:
            U, V, W = result
            # Verify
            T_target = build_target_tensor(n, m, p)
            T_recon = reconstruct_tensor(U, V, W)
            if norm(T_target - T_recon, 'fro') < 1e-8:
                best_result = (rank, U, V, W)
                break  # Found the minimum rank for this starting point
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: use the naive rank-105 decomposition
    naive_rank = n * m * p
    U = np.zeros((n*m, naive_rank))
    V = np.zeros((m*p, naive_rank))
    W = np.zeros((p*n, naive_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
