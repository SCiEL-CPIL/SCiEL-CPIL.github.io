import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    def compute_error(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = sum(W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] for c in range(r))
                        target = 1.0 if j1 == j2 else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err
        return max_err
    
    def try_rank(r):
        max_trials = 300
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(99999 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            err = compute_error(U, V, W, r)
            if err < 0.5:
                # Try to refine
                for it in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Coordinate descent
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            for c in range(r):
                                # Compute gradient w.r.t. U[i*m+j, c]
                                grad = 0
                                for k in range(p):
                                    for j2 in range(m):
                                        target = 1.0 if j == j2 else 0.0
                                        current = sum(W[k*n+i, cc] * U[i*m+j, cc] * V[j2*p+k, cc] for cc in range(r))
                                        # This is getting too slow
                                        pass
                            break
                        break
                    break
                break
            
            if err < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if compute_error(U_r, V_r, W_r, r) < 1e-6:
                    return (U_r, V_r, W_r)
        
        return None
    
    for r in range(4, 20):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Schoolbook fallback
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
