import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Build target tensor
    T = np.zeros((size_U, size_V, size_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def verify_decomposition(U, V, W):
        """Check if decomposition is exact (within float precision)"""
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_hat - T)
        return np.max(diff) < 1e-9
    
    def round_to_half_int(M):
        """Round matrix entries to nearest 0.5"""
        return np.round(M * 2) / 2
    
    def try_rank(R, max_restarts=5):
        """Try to find a valid decomposition of rank R using least_squares"""
        num_vars = (size_U + size_V + size_W) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            np.random.seed(42 + restart * 1000)
            x0 = np.random.randn(num_vars) * 0.1
            
            def residuals(x):
                U = x[:size_U*R].reshape(size_U, R)
                V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                W = x[(size_U+size_V)*R:].reshape(size_W, R)
                
                # Compute full tensor product
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Return flattened difference
                return (T_full - T).flatten()
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                
                if res.cost < 1e-8:
                    x_sol = res.x
                    # Snap to half-integers
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    # Verify exact match after snapping
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    if verify_decomposition(U, V, W):
                        return U.tolist(), V.tolist(), W.tolist()
                    
                    # If snapping fails, try without snapping (might be integer-valued)
                    if verify_decomposition(U, V, W):
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception as e:
                pass
        
        return None
    
    # Try ranks from 14 upwards (known lower bound for this tensor is around 14-16)
    for R in range(14, 20):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(R, max_restarts=3)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank-30
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
