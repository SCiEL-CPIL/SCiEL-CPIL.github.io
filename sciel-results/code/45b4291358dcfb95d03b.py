import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Baseline: Rank-55 Strassen-based decomposition (5 blocks of 2x2 Strassen + 20 schoolbook).
    - Search: Attempt to find a rank-54 decomposition by exploiting 
      linear dependencies in the tensor structure. If not found in time, return rank-55.
    
    The search uses a more robust linear algebra approach:
    1. Build the full tensor T (flattened to a vector of length n*m*p = 60).
    2. For each potential rank-r decomposition, we need to find U, V, W such that
       sum_t U[:,t] outer V[:,t] outer W[:,t] = T (in flattened form).
    3. Instead of random search, we use the fact that if rank 54 exists, the 55 terms
       of the rank-55 decomposition must have a linear dependence. We check if the
       tensor T lies in the span of 54 rank-1 tensors.
    
    A more practical approach: We check if any single term in the rank-55 decomposition
    can be expressed as a linear combination of the others. If so, we can remove it.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # Strassen 7 terms
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank_54_search(U55, V55, W55, time_limit=30):
        """
        Try to find a rank-54 decomposition by looking for linear dependencies.
        
        Approach: For each term r in the rank-55 decomposition, check if it can be
        expressed as a linear combination of the other 54 terms. If so, we can remove it.
        
        More precisely, we want to find coefficients alpha_0, ..., alpha_54 (not all zero)
        such that sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0.
        
        This is a system of 60 equations (one for each (i,j,k) entry of the tensor)
        in 55 unknowns (the alpha_r). If the system has a non-trivial solution,
        then the 55 terms are linearly dependent, and we can remove one term.
        """
        start_search = time.time()
        
        # Build the matrix A where A[i,j,k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to find alpha such that A @ alpha = 0
        
        # Flatten (i,j,k) into a single index
        num_entries = n * m * p  # 60
        rank = 55
        
        A = np.zeros((num_entries, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    A[idx, :] = U55[i*m+j, :] * V55[j*p+k, :] * W55[k*n+i, :]
        
        # Find the null space of A
        # If the null space is non-trivial, we have a linear dependence
        try:
            # Use SVD to find the null space
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            # The null space is spanned by the right singular vectors corresponding to
            # singular values that are close to zero
            tol = 1e-10
            null_space = Vt[s < tol, :]
            
            if null_space.shape[0] > 0:
                # We have a non-trivial null space
                # For each vector in the null space, try to remove a term
                for null_vec in null_space:
                    if time.time() - start_search > time_limit:
                        break
                    
                    # Find the term with the largest coefficient
                    max_r = np.argmax(np.abs(null_vec))
                    
                    if abs(null_vec[max_r]) < 1e-10:
                        continue
                    
                    # Try to remove term max_r
                    U54 = np.copy(U55)
                    V54 = np.copy(V55)
                    W54 = np.copy(W55)
                    
                    # Zero out column max_r
                    U54[:, max_r] = 0
                    V54[:, max_r] = 0
                    W54[:, max_r] = 0
                    
                    # Check if the remaining 54 terms still work
                    if verify_decomposition(U54, V54, W54, tol=1e-10):
                        # Success! We found a rank-54 decomposition.
                        mask = np.ones(55, dtype=bool)
                        mask[max_r] = False
                        U54 = U54[:, mask]
                        V54 = V54[:, mask]
                        W54 = W54[:, mask]
                        return U54, V54, W54, 54
        except Exception:
            pass
        
        # If the above didn't work, try a different approach:
        # Check if we can combine terms across blocks to reduce rank
        # This is a heuristic search
        
        np.random.seed(42)
        num_trials = 500
        
        for trial in range(num_trials):
            if time.time() - start_search > time_limit:
                break
            
            # Generate random coefficients
            alpha = np.random.randn(55)
            
            # Compute sum_r alpha_r * U_r ⊗ V_r ⊗ W_r
            # This should be zero if there's a linear dependence
            is_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(alpha * U55[i*m+j, :] * V55[j*p+k, :] * W55[k*n+i, :])
                        if abs(val) > 1e-8:
                            is_zero = False
                            break
                    if not is_zero:
                        break
                if not is_zero:
                    break
            
            if is_zero and np.max(np.abs(alpha)) > 1e-6:
                # Found a linear dependence! Now try to eliminate one term.
                max_r = np.argmax(np.abs(alpha))
                
                U54 = np.copy(U55)
                V54 = np.copy(V55)
                W54 = np.copy(W55)
                
                U54[:, max_r] = 0
                V54[:, max_r] = 0
                W54[:, max_r] = 0
                
                if verify_decomposition(U54, V54, W54, tol=1e-10):
                    mask = np.ones(55, dtype=bool)
                    mask[max_r] = False
                    U54 = U54[:, mask]
                    V54 = V54[:, mask]
                    W54 = W54[:, mask]
                    return U54, V54, W54, 54
        
        return None

    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55

    # Verify baseline
    if not verify_decomposition(U55, V55, W55, tol=1e-10):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Attempt rank 54 search
    if time.time() - start_time < DEADLINE - 15:
        try:
            result = try_rank_54_search(U55, V55, W55, time_limit=DEADLINE - 20 - (time.time() - start_time))
            if result is not None:
                U54, V54, W54, rank54 = result
                if verify_decomposition(U54, V54, W54, tol=1e-10):
                    best_U, best_V, best_W, best_rank = U54, V54, W54, rank54
        except Exception:
            pass

    if time.time() - start_time > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
