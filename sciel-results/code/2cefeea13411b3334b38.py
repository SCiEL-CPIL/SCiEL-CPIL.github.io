import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def _build_tensor(n, m, p):
    """Build the tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _compute_residual(x, T, n, m, p, R):
    """Compute the residual vector for the least squares problem."""
    # Unpack x into U, V, W
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U = x[:size_U].reshape(n*m, R)
    V = x[size_U:size_U+size_V].reshape(m*p, R)
    W = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
    
    # Compute sum of outer products
    result = np.zeros(T.shape)
    for r in range(R):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    return (result - T).flatten()

def _try_rank_scipy(T, R, n, m, p, max_restarts=5, timeout=10.0):
    """Try to find a decomposition of rank R using scipy.optimize.least_squares."""
    start_time = time.time()
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    total_size = size_U + size_V + size_W
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(42 + restart)
        # Random initialization
        x0 = np.random.randn(total_size) * 0.5
        
        try:
            # Use least_squares
            res = least_squares(
                _compute_residual, 
                x0, 
                args=(T, n, m, p, R), 
                method='lm', # Levenberg-Marquardt, good for least squares
                max_nfev=10000
            )
            
            if res.success or res.cost < 1e-10:
                # Extract U, V, W
                U = res.x[:size_U].reshape(n*m, R)
                V = res.x[size_U:size_U+size_V].reshape(m*p, R)
                W = res.x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exact decomposition with rounded values
                result = np.zeros(T.shape)
                for r in range(R):
                    result += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                
                if np.allclose(result, T, atol=1e-9):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
        except Exception as e:
            continue
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try to find rank 17 and 16
    for R in [17, 16]:
        if time.time() > deadline - 15:
            break
            
        result = _try_rank_scipy(T, R, n, m, p, max_restarts=10, timeout=10.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook rank 18
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
