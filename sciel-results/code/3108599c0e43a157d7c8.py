"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(np.array(U[i*m+j]) * np.array(V[j*p+k]) * np.array(W[k*n+i]))
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank23():
        """Try to find a rank-23 decomposition."""
        rank = 23
        np.random.seed(123)
        
        # Multiple random restarts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            params0 = np.random.randn(27 * rank) * 0.5
            
            def objective(params):
                U = params[:9*rank].reshape(9, rank)
                V = params[9*rank:18*rank].reshape(9, rank)
                W = params[18*rank:27*rank].reshape(9, rank)
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                return error
            
            def gradient(params):
                U = params[:9*rank].reshape(9, rank)
                V = params[9*rank:18*rank].reshape(9, rank)
                W = params[18*rank:27*rank].reshape(9, rank)
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = val - 1.0
                            # d/dU[i*m+j, r] = err * V[j*p+k, r] * W[k*n+i, r]
                            gU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                            gV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                            gW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            result = minimize(objective, params0, jac=gradient, method='BFGS',
                            options={'maxiter': 1000, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U = result.x[:9*rank].reshape(9, rank)
                V = result.x[9*rank:18*rank].reshape(9, rank)
                W = result.x[18*rank:27*rank].reshape(9, rank)
                
                # Try rounding to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # Try rounding to nearest integer
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                
                if verify(U_i, V_i, W_i):
                    return U_i.tolist(), V_i.tolist(), W_i.tolist()
        
        return None
    
    # Try rank 23
    result = try_rank23()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-27
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
