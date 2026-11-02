import numpy as np
import time

DIMENSIONS = [3, 5, 6]
TIME_LIMIT = 110  # Safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a_idx, b_idx, c_idx] = 1 iff a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_school[a_idx, col] = 1.0
                V_school[b_idx, col] = 1.0
                W_school[c_idx, col] = 1.0
                col += 1
    
    # Verify schoolbook decomposition
    def verify_decomposition(U, V, W, T):
        """Verify that sum_r U_r ⊗ V_r ⊗ W_r = T"""
        R = U.shape[1]
        T_approx = np.zeros_like(T)
        for r in range(R):
            T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_approx, T, atol=1e-10)
    
    if not verify_decomposition(U_school, V_school, W_school, T):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a lower rank using structured search
    # For 3x5x6, the known lower bound is around 60-70, but finding exact
    # decompositions is very hard. We'll try a few approaches.
    
    def try_rank(R, max_iters=30, seed=42):
        """Try to find a rank-R decomposition using ALS-like optimization"""
        if time.time() - start_time > TIME_LIMIT - 10:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            # Compute current error
            T_approx = np.zeros_like(T)
            for r in range(R):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T - T_approx)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Update U: for each r, solve linear system
            for r in range(R):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                # Residual without term r
                T_res = T.copy()
                for s in range(R):
                    if s != r:
                        T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                
                # We want to minimize ||T_res - U_r ⊗ V_r ⊗ W_r||
                # This is not linear in U_r, so use gradient step
                # Gradient wrt U[:, r]: (V_r ⊙ W_r) * sum over (b,c) where T_res[a,b,c] contributes
                # Simplified: use a small step in direction that reduces error
                grad_U = np.zeros(n * m)
                for b in range(m * p):
                    for c in range(p * n):
                        if T_res[:, b, c].any():
                            grad_U += V[b, r] * W[c, r] * T_res[:, b, c]
                
                # Normalize and take small step
                grad_norm = np.linalg.norm(grad_U)
                if grad_norm > 1e-10:
                    lr = 0.01
                    U[:, r] -= lr * grad_U / grad_norm
            
            # Update V similarly
            for r in range(R):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                T_res = T.copy()
                for s in range(R):
                    if s != r:
                        T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                
                grad_V = np.zeros(m * p)
                for a in range(n * m):
                    for c in range(p * n):
                        if T_res[a, :, c].any():
                            grad_V += U[a, r] * W[c, r] * T_res[a, :, c]
                
                grad_norm = np.linalg.norm(grad_V)
                if grad_norm > 1e-10:
                    lr = 0.01
                    V[:, r] -= lr * grad_V / grad_norm
            
            # Update W similarly
            for r in range(R):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                T_res = T.copy()
                for s in range(R):
                    if s != r:
                        T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                
                grad_W = np.zeros(p * n)
                for a in range(n * m):
                    for b in range(m * p):
                        if T_res[a, b, :].any():
                            grad_W += U[a, r] * V[b, r] * T_res[a, b, :]
                
                grad_norm = np.linalg.norm(grad_W)
                if grad_norm > 1e-10:
                    lr = 0.01
                    W[:, r] -= lr * grad_W / grad_norm
        
        # Check if we found a valid decomposition
        if best_err < 1e-6:
            if verify_decomposition(best_U, best_V, best_W, T):
                return best_U, best_V, best_W
        
        return None
    
    # Try ranks from 85 down to 70 (limited by time)
    best_result = None
    for R in range(85, 65, -5):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        result = try_rank(R, max_iters=20, seed=R)
        if result is not None:
            best_result = result
            break
    
    # Return the best result found, or fall back to schoolbook
    if best_result is not None:
        U, V, W = best_result
        rank = U.shape[1]
    else:
        U, V, W = U_school, V_school, W_school
        rank = rank_schoolbook
    
    # Final verification
    if not verify_decomposition(U, V, W, T):
        # Fallback to schoolbook if verification fails
        U, V, W = U_school, V_school, W_school
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
