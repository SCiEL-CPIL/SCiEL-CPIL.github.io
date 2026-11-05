import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit

    # Target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Try ranks from 90 down to 88
    # We prioritize lower ranks
    for rank in [90, 89, 88]:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
            def unflatten(x):
                size_U = n * m * rank
                size_V = m * p * rank
                U = x[:size_U].reshape(n * m, rank)
                V = x[size_U:size_U + size_V].reshape(m * p, rank)
                W = x[size_U + size_V:].reshape(p * n, rank)
                return U, V, W
                
            def loss(x):
                U, V, W = unflatten(x)
                T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
                return np.sum((T_comp - T)**2)
                
            from scipy.optimize import minimize
            
            # Use Nelder-Mead for simplicity
            res = minimize(loss, flatten(U, V, W), method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < 1e-3:
                U, V, W = unflatten(res.x)
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check exactness
                T_comp = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_comp, T, atol=1e-9):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
    # Fallback to schoolbook
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
