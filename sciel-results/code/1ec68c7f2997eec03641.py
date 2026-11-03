import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j'*p+k, K=k'*n+i' and j==j', i==i', k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T"""
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False
    
    def schoolbook_decomposition():
        """Return the trivial rank 45 decomposition"""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
                    
        return U, V, W
    
    # Try to find a better decomposition using structured search
    # Strategy: Use alternating least squares to find low-rank CP decomposition
    # Then round to integer/half-integer and verify
    
    def try_rank(r, max_iter=200, tol=1e-8):
        """Try to find a rank-r decomposition using ALS"""
        if time.time() > deadline - 10:
            return None
        
        # Initialize randomly
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            for c in range(r):
                # U[:,c] = T * V[:,c] * W[:,c] / (V[:,c] * W[:,c])
                VW = V[:, c] * W[:, c]
                # Reshape T for matmul
                T_mat = T.reshape(n*m, -1)
                VW_mat = VW.reshape(-1)
                U[:, c] = (T_mat @ VW_mat) / (VW_mat @ VW_mat + 1e-12)
            
            # Update V
            for c in range(r):
                UW = U[:, c] * W[:, c]
                T_mat = T.reshape(m*p, -1)
                UW_mat = UW.reshape(-1)
                V[:, c] = (T_mat @ UW_mat) / (UW_mat @ UW_mat + 1e-12)
            
            # Update W
            for c in range(r):
                UV = U[:, c] * V[:, c]
                T_mat = T.reshape(p*n, -1)
                UV_mat = UV.reshape(-1)
                W[:, c] = (T_mat @ UV_mat) / (UV_mat @ UV_mat + 1e-12)
        
        # Check if we found a good decomposition
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(R - T)
        if err < tol:
            # Try to round to integer/half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            R_rounded = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(R_rounded, T, atol=1e-9):
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(20, 46):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to schoolbook
        U, V, W = schoolbook_decomposition()
    else:
        U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
