import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140  # 140s safety margin
    
    # Schoolbook rank-36 decomposition as baseline
    def make_schoolbook():
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
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Verify decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def objective(params, r, n, m, p):
        """
        Compute the sum of squared errors for the tensor decomposition.
        params: flattened array of U, V, W
        """
        # Unpack
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape((n, m, r))
        V = V_flat.reshape((m, p, r))
        W = W_flat.reshape((p, n, r))
        
        # Compute C[i, k] = sum_j A[i, j] B[j, k]
        # The tensor T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # We want T[i, j, k] == 1 for all i, j, k.
        
        # Efficient computation:
        # For each r, we have outer products.
        # Let's just loop for clarity as dimensions are small.
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i, j, r_idx] * V[j, k, r_idx] * W[k, i, r_idx]
                    error += (s - 1.0) ** 2
        return error

    def try_rank_opt(r):
        """Try to find a rank-r decomposition using random restarts and L-BFGS-B."""
        if time.time() > deadline - 10:
            return None
            
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        total_size = size_U + size_V + size_W
        
        # Random restarts
        num_restarts = 10
        rng = np.random.RandomState(42)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random noise
            params = rng.randn(total_size) * 0.1
            
            # Use L-BFGS-B
            try:
                res = opt.minimize(objective, params, args=(r, n, m, p), method='L-BFGS-B',
                                   options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
                
                if res.success or res.fun < 1e-10:
                    # Reconstruct matrices
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U+size_V]
                    W_flat = res.x[size_U+size_V:size_U+size_V+size_W]
                    
                    U = U_flat.reshape(n*m, r)
                    V = V_flat.reshape(m*p, r)
                    W = W_flat.reshape(p*n, r)
                    
                    # Verify with high precision
                    if verify(U, V, W, tol=1e-8):
                        return U, V, W
            except Exception as e:
                pass
                
        return None

    # Try ranks from 35 down to 32
    # It is unlikely to be lower than 32 for 3x3x4 given Strassen-like bounds, but let's try.
    for r in range(35, 31, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank_opt(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = r
            # If we find a valid one, we can stop or try lower. 
            # Given the difficulty, finding 35 is a good improvement over 36.
            # Let's try to go lower if time permits.
            continue
        else:
            # If we fail at r, it might be possible at r+1? 
            # Usually if rank r is impossible, r-1 is also. But numerical failure doesn't mean impossibility.
            # However, if we fail at 35, 34 might be harder. 
            # Let's break to save time if we fail at a high rank, assuming the landscape is hard.
            # Actually, let's just break. If 35 fails, 34 likely will too or take longer.
            break

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-4):
        # Fall back to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
