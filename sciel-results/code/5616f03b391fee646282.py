"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k]) * 0  # This isn't right
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                val = np.sum(prod)
                max_err = max(max_err, abs(val - 1.0))
    
    return max_err < tol, max_err

def _random_init(n, m, p, r, seed):
    """Random initialization of factor matrices."""
    np.random.seed(seed)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    # Normalize columns
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-15)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-15)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-15)
    return U, V, W

def _gradient_step(U, V, W, n, m, p, lr=0.01):
    """One gradient descent step to minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2"""
    r = U.shape[1]
    
    # Compute gradients
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # val = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                val = np.sum(prod)
                err = val - 1.0
                
                # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
    
    # Gradient descent
    U -= lr * dU
    V -= lr * dV
    W -= lr * dW
    
    return U, V, V

def _try_rank(r, n, m, p, n_restarts=20, time_budget=10):
    """Try to find an exact rank-r decomposition using gradient descent with random restarts."""
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        seed = 42 + restart * 7919
        U, V, W = _random_init(n, m, p, r, seed)
        
        # Run gradient descent
        for it in range(500):
            U, V, W = _gradient_step(U, V, W, n, m, p, lr=0.005)
            
            # Check convergence
            is_exact, err = _verify_decomposition(U, V, W, n, m, p, tol=1e-5)
            if is_exact:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                is_exact_rounded, err_rounded = _verify_decomposition(
                    U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-5)
                if is_exact_rounded:
                    U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n * m)]
                    V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m * p)]
                    W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p * n)]
                    return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Global deadline
    global_start = time.time()
    max_total_time = 110  # safety margin before 150s limit
    
    # Try ranks from low to high
    for r in range(20, 35):
        if time.time() - global_start > max_total_time - 5:
            break
        
        result = _try_rank(r, n, m, p, n_restarts=15, time_budget=7)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
