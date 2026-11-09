import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Schoolbook rank
    R_schoolbook = n * m * p
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def check_solution(U, V, W, R):
        try:
            T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_calc - T)
            return err
        except Exception:
            return float('inf')

    def try_rank(R, num_restarts=5):
        if time.time() > deadline:
            return None
            
        num_vars = R * (n*m + m*p + p*n)
        
        # Use multiple random restarts
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            x0 = np.random.randn(num_vars) * 0.1
            
            def residuals(x):
                if time.time() > deadline:
                    return np.zeros(n*m*m*p*p*n)
                U = x[:R*n*m].reshape(R, n*m).T
                V = x[R*n*m : R*n*m + R*m*p].reshape(R, m*p).T
                W = x[R*n*m + R*m*p :].reshape(R, p*n).T
                T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                return (T_calc - T).flatten()

            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500)
                if result.cost < 1e-10:
                    x = result.x
                    U = x[:R*n*m].reshape(R, n*m).T
                    V = x[R*n*m : R*n*m + R*m*p].reshape(R, m*p).T
                    W = x[R*n*m + R*m*p :].reshape(R, p*n).T
                    
                    # Snap to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    err = check_solution(U_int, V_int, W_int, R)
                    if err < 1e-9:
                        return U_int, V_int, W_int
            except Exception:
                pass
                
        return None

    best_result = None
    # Try ranks 20 to 29
    for R in range(20, R_schoolbook):
        if time.time() > deadline:
            break
        res = try_rank(R, num_restarts=3)
        if res is not None:
            best_result = res
            break
            
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        rank = R_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
