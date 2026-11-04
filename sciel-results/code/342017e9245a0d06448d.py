import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook solution
    rank_schoolbook = n * m * p  # 30
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify_fast(U, V, W, n, m, p, tol=1e-6):
        """Fast verification using tensor contraction."""
        rank = U.shape[1]
        # Check the 30 required entries: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(total - 1) > tol:
                        return False
        # Check that all other entries are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                a = i*m + j
                                b = j2*p + k2
                                c = k*n + i2
                                total = np.sum(U[a] * V[b] * W[c])
                                if abs(total) > tol:
                                    return False
        return True
    
    def round_to_half(x):
        return np.round(x * 2) / 2

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Precompute index arrays for fast residual computation
    required_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                required_indices.append((i*m+j, j*p+k, k*n+i))
    
    zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            a = i*m + j
                            b = j2*p + k2
                            c = k*n + i2
                            zero_indices.append((a, b, c))

    def make_residuals_func(R, fixed_U=None, fixed_V=None, fixed_W=None, fixed_idx=None):
        """
        Create a vectorized residual function for rank R.
        If fixed_* are provided, only optimize the free parameters.
        """
        U_size = n*m*R
        V_size = m*p*R
        W_size = p*n*R
        
        # Determine which parameters are free
        if fixed_U is None:
            free_U = np.arange(n*m*R)
        else:
            free_U = np.array([idx for idx in range(n*m*R) if not (fixed_idx is not None and idx // R in fixed_idx)])
            # This logic is flawed. fixed_idx should be indices of the columns in U, V, W that are fixed.
            # Let's simplify: we fix specific columns.
            # If fixed_U is a matrix, we treat the columns in fixed_idx as fixed.
            # The rest are free.
            # The variable x will contain only the free parameters.
            
            # This is getting complicated. Let's stick to the previous approach but with better init.
            pass
        
        # Re-define residuals_func to handle fixed parameters properly if needed.
        # For now, let's just use the full optimization but with a "peeling" init.
        
        def residuals(x):
            U = x[:U_size].reshape(n*m, R)
            V = x[U_size:U_size+V_size].reshape(m*p, R)
            W = x[U_size+V_size:].reshape(p*n, R)
            
            # If fixed parameters exist, we should have modified U, V, W before calling this.
            # But least_squares passes x directly.
            # So we must ensure that the initial guess x0 has the fixed values,
            # and the optimizer doesn't change them? No, it will change them.
            # To fix parameters, we must reduce the dimension of x.
            
            res = np.empty(len(required_indices) + len(zero_indices))
            
            idx = 0
            for a, b, c in required_indices:
                res[idx] = np.dot(U[a] * V[b], W[c])
                idx += 1
            
            for a, b, c in zero_indices:
                res[idx] = np.dot(U[a] * V[b], W[c])
                idx += 1
            
            res[:len(required_indices)] -= 1.0
            return res
        
        return residuals

    # New Strategy: Rank-1 Peeling with Reduced Optimization
    # We will try to find a rank R decomposition by fixing R-1 terms and optimizing the last one?
    # No, we fix 1 term and optimize R-1 terms.
    
    def try_peeling(R, num_fixed=1, max_restarts=10):
        """
        Try to find a rank R decomposition by fixing 'num_fixed' random rank-1 terms
        and optimizing the remaining R - num_fixed terms.
        """
        U_size = n*m*R
        V_size = m*p*R
        W_size = p*n*R
        
        # Indices for free parameters
        # We fix the first 'num_fixed' columns of U, V, W.
        # The free parameters are the rest.
        
        # Build the residual function that only takes free parameters
        def residuals_free(x_free):
            # Reconstruct U, V, W
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Fill fixed parts
            # The fixed parts are stored in global variables for this closure?
            # Better to pass them or use a non-local variable.
            # Let's use a helper that captures the fixed matrices.
            
            # For simplicity, let's just use the full x and rely on the optimizer
            # staying close to the fixed values if we initialize them well and use a tight tol?
            # No, that's unreliable.
            
            # Let's implement the reduced variable version.
            # x_free contains the free entries.
            # We need to map x_free back to U, V, W.
            
            # This is complex to implement generically in a short time.
            # Alternative: Use the full x, but add a large penalty if the fixed columns change.
            
            U_full = np.zeros((n*m, R))
            V_full = np.zeros((m*p, R))
            W_full = np.zeros((p*n, R))
            
            # This approach is messy.
            return None # Placeholder

        # Let's stick to the full optimization but with a better initialization strategy
        # that mimics peeling:
        # 1. Pick a random rank-1 term.
        # 2. Subtract it from T.
        # 3. Initialize the remaining R-1 terms randomly.
        # 4. Optimize all R terms, but initialize the first term to the random one.
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize U, V, W
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            # "Peeling" Init:
            # Pick a random rank-1 term for the first column
            if num_fixed > 0:
                u1 = np.random.randn(n*m)
                v1 = np.random.randn(m*p)
                w1 = np.random.randn(p*n)
                
                # Scale to be reasonable
                u1 = u1 / np.linalg.norm(u1)
                v1 = v1 / np.linalg.norm(v1)
                w1 = w1 / np.linalg.norm(w1)
                
                U_init[:, 0] = u1
                V_init[:, 0] = v1
                W_init[:, 0] = w1
                
                # Initialize the rest randomly
                U_init[:, 1:] = np.random.randn(n*m, R-1) * 0.5
                V_init[:, 1:] = np.random.randn(m*p, R-1) * 0.5
                W_init[:, 1:] = np.random.randn(p*n, R-1) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            residuals_func = make_residuals_func(R)
            
            try:
                result = least_squares(residuals_func, x0, method='lm', max_nfev=2000)
                U_opt = result.x[:U_size].reshape(n*m, R)
                V_opt = result.x[U_size:U_size+V_size].reshape(m*p, R)
                W_opt = result.x[U_size+V_size:].reshape(p*n, R)
                
                U_rounded = round_to_half(U_opt)
                V_rounded = round_to_half(V_opt)
                W_rounded = round_to_half(W_opt)
                
                if verify_fast(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    return True, U_rounded, V_rounded, W_rounded
            except Exception:
                pass
        
        return False, None, None, None

    np.random.seed(42)
    
    for R in range(rank_schoolbook - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        
        found, U_r, V_r, W_r = try_peeling(R, num_fixed=1, max_restarts=10)
        
        if found:
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            best_rank = R
            break
    
    return best_result
