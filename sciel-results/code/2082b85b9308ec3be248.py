import numpy as np
import scipy.optimize
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    def residuals(x, r):
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = x[:size_u].reshape(n * m, r)
        V = x[size_u:size_u+size_v].reshape(m * p, r)
        W = x[size_u+size_v:].reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        return (T - 1).flatten()
        
    def try_rank(r, max_restarts=10):
        start_time = time.time()
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        total_size = size_u + size_v + size_w
        
        for restart in range(max_restarts):
            if time.time() - start_time > 5.0: # Limit per rank
                break
                
            x0 = np.random.randn(total_size) * 0.5
            try:
                res = scipy.optimize.least_squares(residuals, x0, args=(r,), method='lm', max_nfev=5000)
                
                if res.cost < 1e-6:
                    x_sol = res.x
                    U = x_sol[:size_u].reshape(n * m, r)
                    V = x_sol[size_u:size_u+size_v].reshape(m * p, r)
                    W = x_sol[size_u+size_v:].reshape(p * n, r)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    U_3d_r = U_r.reshape(n, m, r)
                    V_3d_r = V_r.reshape(m, p, r)
                    W_3d_r = W_r.reshape(p, n, r)
                    T_r = np.einsum('ijc,jkc,kic->ijk', U_3d_r, V_3d_r, W_3d_r)
                    
                    if np.allclose(T_r, 1.0, atol=1e-9):
                        return U_r, V_r, W_r
            except Exception:
                pass
                
        return None

    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Try ranks from a lower bound. 
    # Theoretical lower bound is max(n, m, p) = 7? No, for matrix mult it's higher.
    # Let's try 20 to 105.
    for r in range(20, 106):
        if time.time() > deadline:
            break
            
        result = try_rank(r, max_restarts=5)
        if result is not None:
            return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}

    # If no better rank found, return schoolbook
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
