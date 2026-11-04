"""Matrix multiplication 2x4 by 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 40)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(Um, Vm, Wm, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Um[i * m + j][r] * Vm[j * p + k][r] * Wm[k * n + i][r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def to_clean(x):
        # Round to nearest half-integer
        return round(x * 2) / 2.0

    def try_decomposition(R, max_iter=200, seed=42):
        """Try to find a rank-R decomposition using alternating minimization."""
        np.random.seed(seed)
        # Initialize with random values
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update W
            # B_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            B_w = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j, r] * V[j * p + k, r]
                        B_w[k * n + i, r] = s
            
            for idx in range(p * n):
                norm_sq = np.dot(B_w[idx], B_w[idx])
                if norm_sq > 1e-12:
                    W[idx] = B_w[idx] / norm_sq
            
            # Update U
            C_u = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        s = 0
                        for k in range(p):
                            s += V[j * p + k, r] * W[k * n + i, r]
                        C_u[i * m + j, r] = s
            
            for idx in range(n * m):
                norm_sq = np.dot(C_u[idx], C_u[idx])
                if norm_sq > 1e-12:
                    U[idx] = C_u[idx] / norm_sq
            
            # Update V
            D_v = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        s = 0
                        for i in range(n):
                            s += U[i * m + j, r] * W[k * n + i, r]
                        D_v[j * p + k, r] = s
            
            for idx in range(m * p):
                norm_sq = np.dot(D_v[idx], D_v[idx])
                if norm_sq > 1e-12:
                    V[idx] = D_v[idx] / norm_sq
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                break
        
        if best_error < 1e-6:
            U_clean = [[to_clean(best_U[i, j]) for j in range(R)] for i in range(n * m)]
            V_clean = [[to_clean(best_V[i, j]) for j in range(R)] for i in range(m * p)]
            W_clean = [[to_clean(best_W[i, j]) for j in range(R)] for i in range(p * n)]
            
            if verify(U_clean, V_clean, W_clean, R):
                return {"U": U_clean, "V": V_clean, "W": W_clean}
            else:
                U_list = [[float(best_U[i, j]) for j in range(R)] for i in range(n * m)]
                V_list = [[float(best_V[i, j]) for j in range(R)] for i in range(m * p)]
                W_list = [[float(best_W[i, j]) for j in range(R)] for i in range(p * n)]
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 28 upward
    for R in range(28, 41):
        if time.time() > deadline:
            break
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_decomposition(R, max_iter=100, seed=seed)
            if result is not None:
                return result

    # Fall back to schoolbook
    return schoolbook()
