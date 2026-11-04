import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Indices
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Precompute index triples
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
                
    num_eqs = len(indices)
    
    def residual(x, r):
        # x is flattened U, V, W
        # U: n*m*r, V: m*p*r, W: p*n*r
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = x[:size_u].reshape(n * m, r)
        V = x[size_u:size_u + size_v].reshape(m * p, r)
        W = x[size_u + size_v:].reshape(p * n, r)
        
        res = np.zeros(num_eqs)
        for idx, (u_i, v_i, w_i) in enumerate(indices):
            res[idx] = np.dot(U[u_i], V[v_i] * W[w_i]) - 1.0
        return res

    # Try ranks from schoolbook down to 1
    # Start from a reasonable guess, e.g., 32
    # Time limit 150s.
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # We will try a few random restarts for each rank
    # Since the problem is non-convex, we need multiple starts.
    # But time is limited.
    
    for r in range(32, 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try 5 random restarts for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            # Small random values
            np.random.seed(attempt) # Reproducible for debugging, but maybe random is better?
            x0 = np.random.randn(n * m * r + m * p * r + p * n * r) * 0.1
            
            try:
                result = least_squares(residual, x0, args=(r,), method='lm', max_nfev=1000)
                if result.cost < 1e-8:
                    # Check if we can round to integers/half-integers
                    size_u = n * m * r
                    size_v = m * p * r
                    size_w = p * n * r
                    U = result.x[:size_u].reshape(n * m, r)
                    V = result.x[size_u:size_u + size_v].reshape(m * p, r)
                    W = result.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Try rounding to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    res_rounded = residual(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]), r)
                    if np.max(np.abs(res_rounded)) < 1e-6:
                        best_rank = r
                        best_solution = (U_r, V_r, W_r)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        rank = rank_schoolbook
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
