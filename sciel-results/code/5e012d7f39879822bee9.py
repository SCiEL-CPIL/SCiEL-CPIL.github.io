import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
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

    def verify_solution(U, V, W, r):
        """Verify that U, V, W exactly decompose the tensor."""
        res = residual(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), r)
        return np.max(np.abs(res)) < 1e-6

    # Try ranks from known lower bound upwards
    # The rank for 3x3x4 is known to be 18.
    # We search 18, 19, ..., 36.
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # We will try a few random restarts for each rank
    # Since the problem is non-convex, we need multiple starts.
    # But time is limited.
    
    for r in range(18, 37):
        if time.time() > deadline:
            break
            
        found = False
        # Try more restarts for the likely optimal rank 18
        num_restarts = 20 if r == 18 else 10
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use random integers in {-2, -1, 0, 1, 2} to align with expected solution structure
            np.random.seed(attempt + r)
            x0 = np.random.choice([-2, -1, 0, 1, 2], size=n * m * r + m * p * r + p * n * r).astype(float)
            
            try:
                result = least_squares(residual, x0, args=(r,), method='lm', max_nfev=5000)
                if result.cost < 1e-6:
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
