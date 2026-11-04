import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Define indices for the tensor equation:
    # Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    # U shape: (n*m, r) -> index i*m + j
    # V shape: (m*p, r) -> index j*p + k
    # W shape: (p*n, r) -> index k*n + i
    
    def residual(params, r):
        # params: flattened array of length (n*m + m*p + p*n) * r
        U_flat = params[:n*m*r].reshape(n*m, r)
        V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_flat = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_flat[i*m + j]
                    v_val = V_flat[j*p + k]
                    w_val = W_flat[k*n + i]
                    # Sum over r of u_val[r] * v_val[r] * w_val[r]
                    term = np.sum(u_val * v_val * w_val)
                    res[idx] = term - 1.0
                    idx += 1
        return res

    def check_solution(U, V, W):
        # Verify exactness with half-integer rounding
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[i*m+j])):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks from a lower bound up to trivial rank
    # Known theoretical lower bound for M_{2,5,5} is likely around 45.
    # We start at 45 to try to beat the seed (50).
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    for r in range(45, trivial_rank + 1):
        # Number of variables
        n_vars = (n*m + m*p + p*n) * r
        
        # Try a few random restarts for this rank
        found = False
        for _ in range(10):
            # Initialize with small random values
            x0 = np.random.randn(n_vars) * 0.1
            # Add some structure? Maybe start with trivial and zero out?
            # Better: random init
            
            try:
                res = least_squares(residual, x0, args=(r,), method='lm', max_nfev=1000)
                if res.cost < 1e-8:
                    # Extract and round
                    U_flat = res.x[:n*m*r].reshape(n*m, r)
                    V_flat = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_flat = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Convert to lists of floats (or ints if integer)
                    U_list = [[float(val) for val in row] for row in U_rounded]
                    V_list = [[float(val) for val in row] for row in V_rounded]
                    W_list = [[float(val) for val in row] for row in W_rounded]
                    
                    if check_solution(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
        
        # If not found with rounding, try without rounding? 
        # The problem requires integer/half-integer.
        # If numerical solution is very close to half-integers, it should work.
        
    # Fallback to trivial decomposition
    rank = trivial_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
