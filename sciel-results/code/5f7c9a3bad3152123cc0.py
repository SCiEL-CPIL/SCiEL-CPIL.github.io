import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate Baseline (Schoolbook Rank 72)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    def verify(U, V, W, n, m, p):
        """Verify exact decomposition with integer/half-integer tolerance."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def optimize_and_verify(rank, seed, time_limit):
        """
        Attempts to find a rank-r decomposition via L-BFGS-B.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > time_limit:
            return None
            
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Flatten for optimizer
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            # Compute tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We want T[i, j, k] == 1 for all i, j, k
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U_flat[i * m + j, :]
                        v_val = V_flat[j * p + k, :]
                        w_val = W_flat[k * n + i, :]
                        errors[i, j, k] = np.dot(np.dot(u_val, v_val), w_val) - 1.0
            
            return np.sum(errors ** 2)
            
        def gradient(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            dU = np.zeros_like(U_flat)
            dV = np.zeros_like(V_flat)
            dW = np.zeros_like(W_flat)
            
            # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # dT/dU[i*m+j, c] = V[j*p+k, c] W[k*n+i, c]
            # dT/dV[j*p+k, c] = U[i*m+j, c] W[k*n+i, c]
            # dT/dW[k*n+i, c] = U[i*m+j, c] V[j*p+k, c]
            
            # Loss = sum (T[i,j,k] - 1)^2
            # dLoss/dU[i*m+j, c] = sum_{i,j,k} 2(T[i,j,k]-1) * V[j*p+k, c] W[k*n+i, c]
            
            # Vectorized computation for speed
            # We can compute this more efficiently by iterating over c or using einsum, 
            # but for small dimensions and moderate rank, simple loops or batched ops work.
            
            # Let's use einsum for efficiency
            # T = np.einsum('ic,jc,kc->ijk', U_flat, V_flat, W_flat) # This is not quite right because indices are mixed
            # Actually:
            # U_flat is (nm, rank), V_flat is (mp, rank), W_flat is (pn, rank)
            # Index mapping:
            # U[i*m+j, c]
            # V[j*p+k, c]
            # W[k*n+i, c]
            
            # Let's stick to loops for correctness in gradient, or careful einsum.
            # Given small N (3,4,6), loops are fast enough.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # Values
                        u_vec = U_flat[u_idx, :]
                        v_vec = V_flat[v_idx, :]
                        w_vec = W_flat[w_idx, :]
                        
                        t_val = np.dot(np.dot(u_vec, v_vec), w_vec)
                        err = 2.0 * (t_val - 1.0)
                        
                        # Gradients
                        # dU[u_idx, c] += err * v_vec[c] * w_vec[c]
                        dU[u_idx, :] += err * (v_vec * w_vec)
                        dV[v_idx, :] += err * (u_vec * w_vec)
                        dW[w_idx, :] += err * (u_vec * v_vec)
                        
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        except ImportError:
            return None
        except Exception:
            return None

        if time.time() > time_limit:
            return None

        x_opt = result.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Check if optimization converged to a low error
        if result.fun > 1e-6:
            return None
            
        # Round to nearest half-integer
        U_rounded = np.round(U_opt * 2) / 2
        V_rounded = np.round(V_opt * 2) / 2
        W_rounded = np.round(W_opt * 2) / 2
        
        # Verify exactness
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded, V_rounded, W_rounded
            
        return None

    # Try a few ranks below 72
    # Rank 72 is schoolbook. We know 66 is possible. 
    # Let's try 66, 67, 68, etc. 
    # Optimization might be slow, so we limit attempts.
    
    ranks_to_try = [66, 67, 68, 65, 69, 70, 64]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few seeds
        for seed in [42, 123, 999]:
            if time.time() > deadline:
                break
                
            # Set a time limit for this specific optimization call
            # L-BFGS can be slow. Give it 10 seconds per attempt.
            t_limit = time.time() + 10
            if t_limit > deadline:
                t_limit = deadline
                
            res = optimize_and_verify(r, seed, t_limit)
            if res is not None:
                U_r, V_r, W_r = res
                if r < best_rank:
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = r
                # If we found a solution, we might want to try lower ranks immediately
                # but for now, let's just record it.
                # Break seed loop for this rank if we found it? 
                # Actually, different seeds might find different solutions, but rank is fixed.
                # So we can break seed loop if we found a valid decomposition for this rank.
                break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
