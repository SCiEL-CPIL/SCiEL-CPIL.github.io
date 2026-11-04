"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_random_rank(r, seed, max_iters=20000):
        """Try to find rank-r decomposition via random init + gradient descent."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute gradients
            dU = np.zeros((n * m, r))
            dV = np.zeros((m * p, r))
            dW = np.zeros((p * n, r))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                        # d/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                        # d/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                        # The residual for (i,j,k) is: sum_c ... - 1
                        # But we need this to be 1, so gradient of (val-1)^2 / 2 = (val-1) * d(val)/d(param)
                        # Actually, we want sum_c ... = 1, so the "loss" for each (i,j,k) is (sum_c ... - 1)^2
                        # But this couples all (i,j,k). Let's just do a simple projection.
                        
                        pass
            
            # Simpler: use ALS with the correct formulation
            # For fixed V,W: U[i*m+j,:] should satisfy for all k:
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise, shape r)
            # Then U[i*m+j,:] @ M_k = 1 for all k
            # Stack into matrix: A @ U[i*m+j,:]^T = b where A[k,c] = M_k[c], b[k]=1
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-8:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    max_err_r = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                                max_err_r = max(max_err_r, abs(val - 1.0))
                    if max_err_r < 1e-8:
                        return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    # Try various ranks
    for r in [15, 20, 25, 30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_random_rank(r, seed=r * 100 + seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
