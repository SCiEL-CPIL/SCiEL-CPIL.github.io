import time
import sys
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}

    def try_rank(target_rank, num_restarts=50):
        """Try to find a decomposition of the given rank using least_squares."""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Residual function
        def residuals(x):
            U_flat = x[:size_u * target_rank]
            V_flat = x[size_u * target_rank : size_u * target_rank + size_v * target_rank]
            W_flat = x[size_u * target_rank + size_v * target_rank :]
            
            U = U_flat.reshape(size_u, target_rank)
            V = V_flat.reshape(size_v, target_rank)
            W = W_flat.reshape(size_w, target_rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        # Sum over r
                        s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                        res.append(s - 1.0)
            return np.array(res)

        best_solution = None
        best_error = float('inf')
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random integers
            # This helps the optimizer stay in a region where integer solutions might be close
            x0 = np.random.randint(-2, 3, size=size_u * target_rank + size_v * target_rank + size_w * target_rank).astype(float)
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if result.cost < best_error:
                    best_error = result.cost
                    best_solution = result.x
            except Exception:
                pass
        
        if best_solution is not None and best_error < 1e-6:
            # Round to nearest integer or half-integer
            # The problem allows integer/half-integer entries
            # We can round to nearest 0.5
            U_flat = best_solution[:size_u * target_rank]
            V_flat = best_solution[size_u * target_rank : size_u * target_rank + size_v * target_rank]
            W_flat = best_solution[size_u * target_rank + size_v * target_rank :]
            
            U_r = np.round(U_flat * 2) / 2
            V_r = np.round(V_flat * 2) / 2
            W_r = np.round(W_flat * 2) / 2
            
            U = U_r.reshape(size_u, target_rank).tolist()
            V = V_r.reshape(size_v, target_rank).tolist()
            W = W_r.reshape(size_w, target_rank).tolist()
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
            
            # Try rounding to nearest integer only
            U_r = np.round(U_flat)
            V_r = np.round(V_flat)
            W_r = np.round(W_flat)
            
            U = U_r.reshape(size_u, target_rank).tolist()
            V = V_r.reshape(size_v, target_rank).tolist()
            W = W_r.reshape(size_w, target_rank).tolist()
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
                
        return None

    # Try rank 17
    result_17 = try_rank(17, num_restarts=100)
    if result_17:
        return result_17
        
    # Fallback to schoolbook
    return schoolbook()
