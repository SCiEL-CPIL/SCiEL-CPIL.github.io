from scipy.optimize import minimize
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Target equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i in [0, n), j in [0, m), k in [0, p)
    
    def error_func(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:R * size_u].reshape(R, size_u).T  # (size_u, R)
        V = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
        W = params[R * (size_u + size_v) :].reshape(R, size_w).T
        
        # Vectorized computation of the tensor T[i, j, k]
        T = np.zeros((n, m, p))
        
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            w_term = w.T.reshape(n, 1, p)
            
            T += u[:, :, None] * v[None, :, :] * w_term
        
        diff = T - 1.0
        error = np.dot(diff.flatten(), diff.flatten())
        return error

    def round_and_check(U, V, W, R):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_r[:, r].reshape(n, m)
            v = V_r[:, r].reshape(m, p)
            w = W_r[:, r].reshape(p, n)
            w_term = w.T.reshape(n, 1, p)
            T += u[:, :, None] * v[None, :, :] * w_term
        
        if np.allclose(T, 1.0, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, U_r, V_r, W_r

    def try_rank(R, max_attempts=15):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = size_u + size_v + size_w
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization with scaled values
            params = np.random.uniform(-3, 3, R * total_size)
            
            try:
                res = minimize(error_func, params, args=(R,), method='Powell', 
                               options={'maxiter': 300, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                U = res.x[:R * size_u].reshape(R, size_u).T
                V = res.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
                W = res.x[R * (size_u + size_v) :].reshape(R, size_w).T
                
                is_valid, U_r, V_r, W_r = round_and_check(U, V, W, R)
                if is_valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        return None

    # Try ranks from 45 to 59 (below schoolbook 60)
    # Use more attempts for lower ranks
    for R in range(45, 60):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result:
            return result

    # Fallback to schoolbook rank 60
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
