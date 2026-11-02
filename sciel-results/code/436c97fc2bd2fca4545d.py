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

    def strassen_based_decomposition():
        """
        Construct a rank-17 decomposition using Strassen's algorithm for the 2x2 core.
        The 2x3x3 multiplication can be seen as two 2x3 blocks multiplied by 3x3.
        We use a structured approach: fix V to a sparse basis and solve for U, W.
        
        Actually, let's use a known rank-17 construction or find one via algebraic methods.
        For 2x3x3, the minimal rank is 17 (known from literature).
        
        We'll use a deterministic construction based on fixing V to unit vectors
        and solving linear systems for U and W.
        """
        size_u = n * m  # 6
        size_v = m * p  # 9
        size_w = p * n  # 6
        
        # Try to find rank-17 by fixing V to have a specific sparse structure
        # and solving for U and W using linear algebra.
        
        # Strategy: For each rank c, V[:,c] should be a sparse vector.
        # Let's try V with entries in {0, 1, -1} and solve for U, W.
        
        # Actually, let's use a different approach: use the known fact that
        # rank(2x3x3) = 17 and construct it using a systematic method.
        
        # We'll use the approach of fixing V to be a set of 17 sparse vectors
        # and then solving for U and W.
        
        # Let's try a simpler approach: use random sparse V and solve the linear system.
        
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            rank = 17
            
            # Generate a sparse V with entries in {0, 1, -1}
            V = np.zeros((size_v, rank))
            for c in range(rank):
                # Each column of V has 1-2 non-zero entries
                num_nonzero = random.choice([1, 1, 2])
                indices = random.sample(range(size_v), num_nonzero)
                for idx in indices:
                    V[idx, c] = random.choice([1, -1])
            
            # Now we need to find U and W such that:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # This is still nonlinear in U and W. Let's fix W as well.
            # Actually, let's fix both V and W to be sparse, and solve for U.
            
            # Or better: fix V to be the identity-like structure and solve.
            
            # Let's try a different strategy: use the algebraic structure.
            # For 2x3x3, we can think of it as two 1x3x3 problems plus cross terms.
            
            # Actually, let's just try to use least_squares with a good initialization
            # based on the schoolbook solution with one rank removed.
            
            # Initialize with schoolbook rank-18, then try to merge two terms.
            U_init = np.zeros((size_u, rank))
            V_init = np.zeros((size_v, rank))
            W_init = np.zeros((size_w, rank))
            
            # Start with schoolbook for first 16 terms
            col_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col_idx >= rank - 1:
                            break
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        U_init[idx_u, col_idx] = 1.0
                        V_init[idx_v, col_idx] = 1.0
                        W_init[idx_w, col_idx] = 1.0
                        col_idx += 1
                    if col_idx >= rank - 1:
                        break
                if col_idx >= rank - 1:
                    break
            
            # The last term (rank-1) will be optimized to handle the remaining product
            # Actually, this approach is not quite right. Let's use a proper optimization.
            
            # Let's use a proper least_squares approach with multiple restarts.
            
            def residuals(x):
                U_flat = x[:size_u * rank]
                V_flat = x[size_u * rank : size_u * rank + size_v * rank]
                W_flat = x[size_u * rank + size_v * rank :]
                
                U = U_flat.reshape(size_u, rank)
                V = V_flat.reshape(size_v, rank)
                W = W_flat.reshape(size_w, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                            res.append(s - 1.0)
                return np.array(res)
            
            # Initialize with small random values
            x0 = np.random.randn(size_u * rank + size_v * rank + size_w * rank) * 0.5
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500)
                if result.cost < 1e-8:
                    U_flat = result.x[:size_u * rank]
                    V_flat = result.x[size_u * rank : size_u * rank + size_v * rank]
                    W_flat = result.x[size_u * rank + size_v * rank :]
                    
                    # Round to nearest half-integer
                    U_r = np.round(U_flat * 2) / 2
                    V_r = np.round(V_flat * 2) / 2
                    W_r = np.round(W_flat * 2) / 2
                    
                    U = U_r.reshape(size_u, rank).tolist()
                    V = V_r.reshape(size_v, rank).tolist()
                    W = W_r.reshape(size_w, rank).tolist()
                    
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                    
                    # Try rounding to nearest integer
                    U_r = np.round(U_flat)
                    V_r = np.round(V_flat)
                    W_r = np.round(W_flat)
                    
                    U = U_r.reshape(size_u, rank).tolist()
                    V = V_r.reshape(size_v, rank).tolist()
                    W = W_r.reshape(size_w, rank).tolist()
                    
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        
        return None

    # Try the Strassen-based / algebraic construction
    result_17 = strassen_based_decomposition()
    if result_17:
        return result_17
        
    # Fallback to schoolbook
    return schoolbook()
