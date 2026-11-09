"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def to_half_integer(x):
        """Round to nearest 0.5"""
        return np.round(2.0 * x) / 2.0
        
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
        
    def verify_strict(U, V, W, n, m, p):
        """Check if decomposition is exact with strict integer/half-integer check"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return verify(U, V, W, n, m, p, tol=1e-12)
    
    def compute_residual(U, V, W, n, m, p):
        """Compute total residual"""
        rank = U.shape[1]
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    residual += (s - 1.0) ** 2
        return residual
    
    def als_iteration(U, V, W, n, m, p, r):
        """One ALS iteration"""
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-14)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-14)[0]
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-14)[0]
                except:
                    pass
        return U, V, W
    
    def polish(U, V, W, n, m, p, r, max_iters=50, tol=1e-14):
        """Polish with tighter tolerance"""
        prev_res = compute_residual(U, V, W, n, m, p)
        for _ in range(max_iters):
            U, V, W = als_iteration(U, V, W, n, m, p, r)
            res = compute_residual(U, V, W, n, m, p)
            if abs(prev_res - res) < tol * max(1.0, prev_res):
                break
            prev_res = res
        return U, V, W
    
    def try_rank(r, seed, time_limit):
        """Try to find a rank-r decomposition"""
        start = time.time()
        np.random.seed(seed)
        
        # Multiple initializations
        best_U, best_V, best_W = None, None, None
        best_res = float('inf')
        
        n_init = 10
        for init in range(n_init):
            if time.time() - start > time_limit:
                break
                
            # Different initialization strategies
            if init == 0:
                # Random
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
            elif init == 1:
                # Structured: based on schoolbook pattern
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for col in range(r):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n * m)) % p
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                # Add small random perturbation
                U += np.random.randn(n * m, r) * 0.1
                V += np.random.randn(m * p, r) * 0.1
                W += np.random.randn(p * n, r) * 0.1
            elif init == 2:
                # Kronecker-like structure
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
                # Scale to have unit norm columns
                for c in range(r):
                    norm = np.linalg.norm(U[:, c])
                    if norm > 1e-10:
                        U[:, c] /= norm
                        V[:, c] *= norm
                        W[:, c] *= norm
            else:
                U = np.random.randn(n * m, r) * (0.2 + 0.1 * init)
                V = np.random.randn(m * p, r) * (0.2 + 0.1 * init)
                W = np.random.randn(p * n, r) * (0.2 + 0.1 * init)
            
            # Run ALS
            for iteration in range(200):
                if time.time() - start > time_limit:
                    break
                U, V, W = als_iteration(U, V, W, n, m, p, r)
            
            res = compute_residual(U, V, W, n, m, p)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_res < 1e-10:
                break
        
        if best_U is None:
            return None
            
        # Polish
        best_U, best_V, best_W = polish(best_U, best_V, best_W, n, m, p, r)
        res = compute_residual(best_U, best_V, best_W, n, m, p)
        
        if res < 1e-10:
            # Snap to half-integers
            U_h = to_half_integer(best_U)
            V_h = to_half_integer(best_V)
            W_h = to_half_integer(best_W)
            
            if verify_strict(U_h, V_h, W_h, n, m, p):
                return U_h, V_h, W_h
            
            # If snapping failed, try without snapping (shouldn't happen for valid half-int)
            if verify(best_U, best_V, best_W, n, m, p, tol=1e-12):
                return best_U, best_V, best_W
                
        return None
    
    best_result = None
    
    # Try ranks from low to high
    # Lower bound: at least n*m = 6 (since each of the 6 output entries needs to be formed)
    # Realistic: try 27 (block structure), then 26, 25, ...
    # Theoretical minimum for 2x3x9: at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Actually the tensor rank lower bound is more subtle. Let's try from 27 down.
    
    for r in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 15:
            break
            
        remaining = deadline - time.time()
        time_per_rank = min(10.0, remaining / max(1, (27 - r + 1)))
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            time_per_try = min(3.0, (deadline - 10 - time.time()) / max(1, 5 - seed))
            result = try_rank(r, seed, time_per_try)
            if result is not None:
                U_h, V_h, W_h = result
                if verify_strict(U_h, V_h, W_h, n, m, p):
                    best_result = (r, U_h, V_h, W_h)
                    break
        if best_result is not None:
            break
            
    if best_result is not None:
        r, U_h, V_h, W_h = best_result
        return {
            "U": U_h.tolist(),
            "V": V_h.tolist(),
            "W": W_h.tolist()
        }
        
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
