import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0

    # Target Tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def flatten_matrices(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_matrices(x, rank):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        U = x[:size_u].reshape(n * m, rank)
        V = x[size_u:size_u + size_v].reshape(m * p, rank)
        W = x[size_u + size_v:size_u + size_v + size_w].reshape(p * n, rank)
        return U, V, W

    def residuals(x, rank):
        U, V, W = unflatten_matrices(x, rank)
        # Compute sum U_l V_l W_l
        # This is a tensor contraction
        # Res[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
        # We can compute this efficiently
        # Res = np.einsum('al,bl,cl->abc', U, V, W)
        Res = np.einsum('al,bl,cl->abc', U, V, W)
        # Flatten the residual
        return (Res - T).flatten()

    # Try ranks from 26 down to 23
    # Start with 26 as it's most likely to be found quickly if it exists
    for rank in [26, 25, 24, 23]:
        if time.time() > deadline - 20:
            break
            
        # Initialize
        # Strategy: Random initialization
        # Or perturbed schoolbook?
        # For rank 26, we can't directly use schoolbook (27 terms).
        # We can try to initialize with random small values.
        
        # Better initialization: 
        # If we had a rank 27 solution, we could try to drop one term and optimize.
        # Let's try random initialization with a few restarts.
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            # Random init
            x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank)
            
            try:
                sol = least_squares(residuals, x0, args=(rank,), method='lm', max_nfev=1000)
                if sol.success or np.linalg.norm(sol.fun) < 1e-6:
                    U, V, W = unflatten_matrices(sol.x, rank)
                    # Verify
                    Res = np.einsum('al,bl,cl->abc', U, V, W)
                    if np.allclose(Res, T, atol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except Exception:
                pass

    # Fallback to rank 27
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
