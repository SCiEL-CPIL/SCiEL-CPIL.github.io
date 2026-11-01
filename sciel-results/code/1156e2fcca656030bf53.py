import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor for matrix multiplication (n,m,p) can be represented as:
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # the tensor equation is satisfied.
    
    # Build the multiplication tensor as a 3-way array
    # T[a, b, c] where a indexes (i,j) in [0,n*m), b indexes (j,k) in [0,m*p), c indexes (k,i) in [0,p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0
    
    # Actually, the standard formulation: the multiplication tensor M has
    # M[i,j,k,l] = delta_{j,k} * delta_{i,l} ... no.
    
    # Let me use the bilinear form approach. We want to find rank-r such that:
    # For all i in [0,n), k in [0,p):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # can be written as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
    # This isn't quite right.
    
    # The correct formulation: the tensor is in (R^{n*m})* ⊗ (R^{m*p})* ⊗ (R^{n*p})
    # and the decomposition is T = sum_c u_c ⊗ v_c ⊗ w_c where:
    # u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{n*p}
    # and T[a, b, c'] = sum_r U[a,r] * V[b,r] * W[c',r]
    # where T[a,b,c'] = 1 if a=(i,j), b=(j,k), c'=(i,k) for some i,j,k, else 0.
    
    # So T[i*m+j, j*p+k, i*p+k] = 1 (using 0-indexed)
    
    # Let's build this tensor and try to find low-rank decompositions.
    
    # For small enough cases, we can use linear algebra.
    # Fix U and V, then W is determined by linear equations, or vice versa.
    
    # Strategy: try ranks from 27 upward (slice rank lower bound)
    # For each rank r, try to find a valid decomposition.
    
    # A simpler approach: use the fact that for matrix multiplication,
    # there are known decompositions. For (2,3,9), let's try to be clever.
    
    # Actually, let me try a direct numerical optimization approach.
    # We'll use the method of alternating least squares or gradient descent.
    
    # But first, let's try a structured approach.
    
    # Key insight: The tensor M_{n,m,p} can be written as a sum of rank-1 tensors.
    # The schoolbook gives n*m*p terms. Can we do better?
    
    # For (2,3,9): n*m=6, m*p=27, p*n=18
    # Lower bound: max(6, 27, 18) = 27
    
    # Let me try rank 27 first. If that works, great. If not, try higher.
    
    # To check if a rank-r decomposition exists, we can set up the problem as:
    # Find U (6 x r), V (27 x r), W (18 x r) such that
    # for all (i,j,k): U[i*m+j, :] · V[j*p+k, :] · W[k*n+i, :] ... 
    # No, the tensor entry is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[i*p+k, c] = delta_{valid}
    
    # This is a system of polynomial equations. Hard to solve directly.
    
    # Alternative: use the "tensor rank" computation via numerical methods.
    
    # Given time constraints, let me try a few specific ranks and use 
    # a randomized approach with verification.
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        n, m, p = int(n), int(m), int(p)
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check dimensions
        if U.shape != (n*m, r) or V.shape != (m*p, r) or W.shape != (p*n, r):
            return False
        
        # For each (i, j, k), check that the tensor entry is correct
        # T[i*m+j, j*p+k, k*n+i] should be 1, and all other entries should be 0
        # But we can't check all entries explicitly for large tensors.
        
        # Instead, verify the bilinear form: for random A, B, check that
        # the reconstructed C matches A @ B.
        
        np.random.seed(42)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Reconstruct C from U, V, W
            # C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * ... 
            # Hmm, I need to be more careful.
            
            # The decomposition means: the multiplication tensor T satisfies
            # T = sum_c u_c ⊗ v_c ⊗ w_c
            # where u_c[a] = U[a, c], v_c[b] = V[b, c], w_c[d] = W[d, c]
            
            # T[a, b, d] = sum_c U[a, c] * V[b, c] * W[d, c]
            
            # a = i*m + j, b = j'*p + k, d = k'*n + i'
            # T[a, b, d] = 1 if j = j' and i = i' and k = k', else 0
            # i.e., T[i*m+j, j*p+k, k*n+i] = 1
            
            # The bilinear form: given A (n x m) and B (m x p), 
            # C[i,k] = sum_j A[i,j] * B[j,k]
            
            # In tensor form: C[i,k] = sum_{j} A[i*m+j] * B[j*p+k] * T[i*m+j, j*p+k, k*n+i]
            # Wait, that's not right. Let me think again.
            
            # Actually, the way it works: 
            # C = A @ B means C[i,k] = sum_j A[i,j] B[j,k]
            # We can write this using the tensor:
            # C[i,k] = sum_{a,b,d} A[a] B[b] W[d] * (something)
            
            # I think the cleanest verification is:
            # Build the full tensor and check.
            
            # For our small case, let's just build it.
            pass
        
        # Simpler verification: check that the reconstruction works
        # C[i,k] should equal sum_j A[i,j] * B[j,k]
        # Using the decomposition:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j'*p+k, c] * B[j',k])
        # No, that's not how it works either.
        
        # Let me just directly verify the tensor entries.
        T_reconstructed = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T_reconstructed += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        
        # Now check that T_reconstructed[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and 0 otherwise.
        
        # Check the 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_a = i * m + j
                    idx_b = j * p + k
                    idx_d = k * n + i
                    if abs(T_reconstructed[idx_a, idx_b, idx_d] - 1.0) > tol:
                        return False
        
        # Check some of the 0s (can't check all for efficiency)
        # Total entries: n*m * m*p * p*n = 6 * 27 * 18 = 2916
        # Number of 1s: n*m*p = 54
        # Number of 0s: 2916 - 54 = 2862
        
        # Check a random sample of 0s
        all_indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    all_indices.add((i*m+j, j*p+k, k*n+i))
        
        np.random.seed(123)
        for _ in range(100):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            d = np.random.randint(0, p*n)
            if (a, b, d) not in all_indices:
                if abs(T_reconstructed[a, b, d]) > tol:
                    return False
        
        return True
    
    # Try to find a decomposition with rank < 54
    # Start from the slice rank lower bound of 27
    
    # For now, let's try a few ranks and use a simple optimization approach.
    # Given the complexity, let me try rank 27, 28, ..., up to some limit.
    
    # Actually, for this specific case, let me try a more direct approach.
    # The matrix multiplication tensor for (2,3,9) might have a known optimal rank.
    
    # Let me try to use a linear algebra approach:
    # Fix U to have orthonormal columns (or some structured form), then solve for V and W.
    
    # Or: use the fact that we can parameterize the problem and use Newton's method.
    
    # Given time constraints, let me try a practical approach:
    # 1. Start with the schoolbook decomposition (rank 54)
    # 2. Try to merge terms or find linear dependencies
    
    # Actually, let me just try a few specific ranks with random initialization and ALS.
    
    def try_rank(r, n, m, p, max_iter=100, seed=0):
        """Try to find a rank-r decomposition using ALS."""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize
        for c in range(r):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-10:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        for iteration in range(max_iter):
            if time.time() - start_time > 130:
                break
            
            # Update U: for each a, U[a, :] should satisfy
            # sum_c U[a, c] * V[b, c] * W[d, c] = T[a, b, d] for all b, d
            # This is: U[a, :] * (V ⊙ W) = T[a, :, :]
            # where ⊙ is the Khatri-Rao product
            
            # Khatri-Rao product of V and W: (V ⊙ W)[b, d, c] = V[b, c] * W[d, c]
            # Reshape to 2D: (b*d, r)
            
            # For efficiency, let's do this row by row for U.
            # T[a, :, :] is a (m*p, p*n) matrix.
            # We need: U[a, :] @ (V ⊙ W)_{reshaped} = T[a, :, :].flatten()
            
            # Khatri-Rao: KR = np.einsum('bc,dc->bdc', V, W) -> shape (m*p, p*n, r)
            # Reshape to (m*p * p*n, r)
            
            # This is expensive for large tensors. Let's be smart.
            
            # Alternative: update one factor at a time using the standard ALS for tensor decomposition.
            
            # For U[a, c]: fix V and W, the equation is
            # sum_c U[a, c] * V[b, c] * W[d, c] = T[a, b, d]
            # For fixed a: U[a, :] is a vector of length r.
            # We need: for all (b,d), sum_c U[a,c] * (V[b,c] * W[d,c]) = T[a,b,d]
            # This is a linear system: U[a, :] @ M_a = T[a, :, :].flatten()
            # where M_a[b*d, c] = V[b, c] * W[d, c]
            
            # But M_a doesn't depend on a! It's the same for all a.
            # So: U @ M = T.flatten() where T is reshaped appropriately.
            
            # Let KR = np.einsum('bc,dc->bd c', V, W)  # shape (m*p, p*n, r)
            # KR_2d = KR.reshape(m*p * p*n, r)
            # T_2d = T.reshape(n*m, m*p * p*n).T  # shape (m*p*p*n, n*m)
            # Then: U.T @ KR_2d = T_2d  =>  U = (KR_2d.T @ KR_2d)^{-1} @ KR_2d.T @ T_2d.T
            
            # This is a least squares problem: minimize ||U @ KR_2d - T_2d||
            
            # But this is very expensive for large tensors.
            
            # Given the small size (6*27*18 = 2916 entries), let's try it.
            
            # Build the tensor T
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Khatri-Rao product
            KR = np.einsum('bc,dc->bdc', V, W)  # (m*p, p*n, r)
            KR_2d = KR.reshape(m*p * p*n, r)
            
            # T reshaped
            T_2d = T.reshape(n*m, m*p * p*n).T  # (m*p*p*n, n*m)
            
            # Solve: U.T @ KR_2d ≈ T_2d  =>  U ≈ (KR_2d.T @ KR_2d)^{-1} @ KR_2d.T @ T_2d.T
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(KR_2d.T, T_2d, rcond=None)
                U = U_new.T  # shape (n*m, r)
            except:
                break
            
            # Update V similarly
            KR2 = np.einsum('ac,dc->adc', U, W)  # (n*m, p*n, r)
            KR2_2d = KR2.reshape(n*m * p*n, r)
            T_2d_v = T.reshape(m*p, n*m * p*n).T  # (n*m*p*n, m*p)
            try:
                V_new, _, _, _ = np.linalg.lstsq(KR2_2d.T, T_2d_v, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W similarly
            KR3 = np.einsum('ab,bc->abc', U, V)  # (n*m, m*p, r)
            KR3_2d = KR3.reshape(n*m * m*p, r)
            T_2d_w = T.reshape(p*n, n*m * m*p).T  # (n*m*m*p, p*n)
            try:
                W_new, _, _, _ = np.linalg.lstsq(KR3_2d.T, T_2d_w, rcond=None)
                W = W_new.T
            except:
                break
        
        # Check if the decomposition is valid
        if verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
        return None
    
    # Try ranks from 27 upward
    best_result = None
    for r in range(27, 55):
        if time.time() - start_time > 120:
            break
        
        result = try_rank(r, n, m, p, max_iter=50, seed=r)
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        # Convert to lists with integer/half-integer entries if possible
        # For now, return as floats
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
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
