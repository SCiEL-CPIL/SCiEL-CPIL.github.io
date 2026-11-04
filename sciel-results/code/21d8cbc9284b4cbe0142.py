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
        # T[a,b,c] where a=i*m+j, b=j'*p+k', c=k''*n+i'
        # T is zero unless i==i', j==j', k==k'
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
    # Required entries: (i, j, k) -> indices (i*m+j, j*p+k, k*n+i)
    required_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                required_indices.append((i*m+j, j*p+k, k*n+i))
    
    # All zero entries
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
    
    def make_residuals_func(R):
        """Create a vectorized residual function for rank R."""
        U_size = n*m*R
        V_size = m*p*R
        
        def residuals(x):
            U = x[:U_size].reshape(n*m, R)
            V = x[U_size:U_size+V_size].reshape(m*p, R)
            W = x[U_size+V_size:].reshape(p*n, R)
            
            res = np.empty(len(required_indices) + len(zero_indices))
            
            # Required entries
            idx = 0
            for a, b, c in required_indices:
                res[idx] = np.dot(U[a] * V[b], W[c])
                idx += 1
            
            # Zero entries
            for a, b, c in zero_indices:
                res[idx] = np.dot(U[a] * V[b], W[c])
                idx += 1
            
            # Set targets: first 30 should be 1, rest should be 0
            res[:len(required_indices)] -= 1.0
            return res
        
        return residuals
    
    # Try lower ranks with multiple restarts
    np.random.seed(42)
    
    for R in range(rank_schoolbook - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        
        residuals_func = make_residuals_func(R)
        U_size = n*m*R
        V_size = m*p*R
        
        found = False
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Perturbed schoolbook
                U_init = U_sb[:, :R].copy()
                V_init = V_sb[:, :R].copy()
                W_init = W_sb[:, :R].copy()
                U_init += 0.1 * np.random.randn(n*m, R)
                V_init += 0.1 * np.random.randn(m*p, R)
                W_init += 0.1 * np.random.randn(p*n, R)
            elif restart == 1:
                # Random initialization with smaller scale
                U_init = 0.5 * np.random.randn(n*m, R)
                V_init = 0.5 * np.random.randn(m*p, R)
                W_init = 0.5 * np.random.randn(p*n, R)
            elif restart == 2:
                # Scaled random
                U_init = 1.0 * np.random.randn(n*m, R)
                V_init = 1.0 * np.random.randn(m*p, R)
                W_init = 1.0 * np.random.randn(p*n, R)
            else:
                # Mix of schoolbook and random
                U_init = 0.5 * U_sb[:, :R] + 0.5 * np.random.randn(n*m, R)
                V_init = 0.5 * V_sb[:, :R] + 0.5 * np.random.randn(m*p, R)
                W_init = 0.5 * W_sb[:, :R] + 0.5 * np.random.randn(p*n, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals_func, x0, method='lm', max_nfev=5000)
                U_opt = result.x[:U_size].reshape(n*m, R)
                V_opt = result.x[U_size:U_size+V_size].reshape(m*p, R)
                W_opt = result.x[U_size+V_size:].reshape(p*n, R)
                
                # Round to half-integers
                U_rounded = round_to_half(U_opt)
                V_rounded = round_to_half(V_opt)
                W_rounded = round_to_half(W_opt)
                
                if verify_fast(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = R
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result
