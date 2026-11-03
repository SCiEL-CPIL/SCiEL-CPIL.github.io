<<<<<<< SEARCH
def _try_rank_reduction(U, V, W, n, m, p, start_rank, min_rank, deadline):
    """
    Attempts to reduce rank by solving linear systems for subsets of terms.
    Uses QR with pivoting to find independent terms.
    """
    target = np.ones(n * m * p)
    
    # Pre-compute tensor vectors for the current U, V, W
    M_full = _get_tensor_vectors(U, V, W, n, m, p)
    current_rank = U.shape[1]
    
    best_result = None
    best_rank = current_rank
    
    # We will try to find a subset of terms of size R that can form the target.
    # We iterate R downwards from start_rank to min_rank.
    for R in range(start_rank, min_rank - 1, -1):
        if time.time() > deadline:
            break
            
        # If R >= current_rank, we already have a solution (or can trivially find one).
        # We are interested in R < current_rank.
        if R >= current_rank:
            continue
            
        # Strategy: Use QR decomposition with column pivoting on M_full to find
        # the R most independent columns.
        # Note: M_full is (80, current_rank). current_rank is ~65.
        # If R < 80, we can check if the target is in the span of these R columns.
        
        try:
            # QR with pivoting
            # M_full * P = Q * R
            # The first R columns of M_full * P are the most independent.
            Q, Rmat, P = np.linalg.qr(M_full, mode='complete')
            # P is a permutation matrix. We need the indices.
            # np.linalg.qr does not return permutation indices directly in all versions?
            # Actually, scipy.linalg.qr does, but we only have numpy.
            # We can use a simple greedy selection or just assume the first R columns of Q
            # span the subspace. But we need the *original* indices to construct U, V, W.
            
            # Let's use a different approach:
            # Randomly select R columns from M_full, solve, check residual.
            # Repeat a few times.
            
            # Since current_rank is small (~65), we can be a bit exhaustive or use greedy.
            # Greedy: Start with empty set, add column that reduces residual most.
            # This is O(R * current_rank * num_entries).
            
            # Let's stick to a simpler heuristic:
            # Try removing one term at a time from the current best, solving for the rest.
            # This is what the previous program tried.
            
            # Better: Try to find a subset of size R from the current R+1 terms.
            # If we have a valid solution of rank R+1, can we drop one term?
            # We can try to express the target as a combination of R terms from the R+1.
            
            # Let's try a random search for subsets of size R.
            # Number of attempts: 20
            for attempt in range(20):
                if time.time() > deadline:
                    break
                    
                # Select R random indices from current_rank
                indices = np.random.choice(current_rank, R, replace=False)
                
                # Extract sub-matrix
                M_sub = M_full[:, indices]
                
                # Solve M_sub * c = target
                # Use lstsq
                c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                
                # Check residual
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # We have a solution!
                    # Construct new U, V, W
                    U_new = np.zeros((n*m, R))
                    V_new = np.zeros((m*p, R))
                    W_new = np.zeros((p*n, R))
                    
                    for idx, orig_idx in enumerate(indices):
                        U_new[:, idx] = U[:, orig_idx] * c[idx]
                        V_new[:, idx] = V[:, orig_idx] * c[idx]
                        W_new[:, idx] = W[:, orig_idx] * c[idx]
                    
                    # Verify
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        # Check half-integer constraint
                        if _verify_integer_half_integer(U_new, V_new, W_new, tol=1e-8):
                            return U_new, V_new, W_new
                        else:
                            # Try rounding
                            U_rounded = np.round(U_new * 2) / 2.0
                            V_rounded = np.round(V_new * 2) / 2.0
                            W_rounded = np.round(W_new * 2) / 2.0
                            
                            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                                return U_rounded, V_rounded, W_rounded
                            
        except Exception:
            pass
            
    return None
=======
def _riemannian_polish(U, V, W, n, m, p, max_iter=500, tol=1e-10, deadline=None):
    """
    Riemannian gradient descent on the Stiefel manifold to polish the decomposition.
    We optimize over scaled versions of U, V, W to minimize the tensor error.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    # Compute current error
    def compute_error():
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    err += (total - 1.0) ** 2
        return err
    
    initial_err = compute_error()
    if initial_err < tol:
        return U, V, W, initial_err
    
    # Riemannian gradient descent
    lr = 0.01
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = initial_err
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    err_term = 2 * (total - 1.0)
                    
                    # d/dU
                    grad_U[i*m+j, :] += err_term * V[j*p+k, :] * W[k*n+i, :]
                    # d/dV
                    grad_V[j*p+k, :] += err_term * U[i*m+j, :] * W[k*n+i, :]
                    # d/dW
                    grad_W[k*n+i, :] += err_term * U[i*m+j, :] * V[j*p+k, :]
        
        # Riemannian gradient projection (Euclidean for now, can improve)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Project back to maintain structure (optional)
        # For now, just clip to reasonable values
        U = np.clip(U, -10, 10)
        V = np.clip(V, -10, 10)
        W = np.clip(W, -10, 10)
        
        current_err = compute_error()
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr = max(0.001, lr * 0.9)
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank_reduction(U, V, W, n, m, p, start_rank, min_rank, deadline):
    """
    Attempts to reduce rank by solving linear systems for subsets of terms.
    Uses QR with pivoting to find independent terms, plus Riemannian polish.
    """
    target = np.ones(n * m * p)
    
    # Pre-compute tensor vectors for the current U, V, W
    M_full = _get_tensor_vectors(U, V, W, n, m, p)
    current_rank = U.shape[1]
    
    best_result = None
    best_rank = current_rank
    
    # We will try to find a subset of terms of size R that can form the target.
    # We iterate R downwards from start_rank to min_rank.
    for R in range(start_rank, min_rank - 1, -1):
        if time.time() > deadline:
            break
            
        # If R >= current_rank, we already have a solution (or can trivially find one).
        # We are interested in R < current_rank.
        if R >= current_rank:
            continue
            
        # Try multiple random subsets
        num_attempts = 30
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            # Select R random indices from current_rank
            indices = np.random.choice(current_rank, R, replace=False)
            
            # Extract sub-matrix
            M_sub = M_full[:, indices]
            
            # Solve M_sub * c = target
            # Use lstsq
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                
                # Check residual
                if len(residuals) > 0 and residuals[0] < 1e-6:
                    # We have a solution!
                    # Construct new U, V, W
                    U_new = np.zeros((n*m, R))
                    V_new = np.zeros((m*p, R))
                    W_new = np.zeros((p*n, R))
                    
                    for idx, orig_idx in enumerate(indices):
                        U_new[:, idx] = U[:, orig_idx] * c[idx]
                        V_new[:, idx] = V[:, orig_idx] * c[idx]
                        W_new[:, idx] = W[:, orig_idx] * c[idx]
                    
                    # Try Riemannian polish
                    U_pol, V_pol, W_pol, err = _riemannian_polish(
                        U_new, V_new, W_new, n, m, p, 
                        max_iter=100, tol=1e-10, deadline=deadline
                    )
                    
                    # Verify
                    if _verify_decomposition(U_pol, V_pol, W_pol, n, m, p, tol=1e-8):
                        # Check half-integer constraint
                        if _verify_integer_half_integer(U_pol, V_pol, W_pol, tol=1e-8):
                            return U_pol, V_pol, W_pol
                        else:
                            # Try rounding
                            U_rounded = np.round(U_pol * 2) / 2.0
                            V_rounded = np.round(V_pol * 2) / 2.0
                            W_rounded = np.round(W_pol * 2) / 2.0
                            
                            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                                return U_rounded, V_rounded, W_rounded
                    elif _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        # Original solution works
                        if _verify_integer_half_integer(U_new, V_new, W_new, tol=1e-8):
                            return U_new, V_new, W_new
                        else:
                            U_rounded = np.round(U_new * 2) / 2.0
                            V_rounded = np.round(V_new * 2) / 2.0
                            W_rounded = np.round(W_new * 2) / 2.0
                            
                            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                                return U_rounded, V_rounded, W_rounded
            except Exception:
                pass
    
    return None
>>>>>>> REPLACE

