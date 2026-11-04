import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = rank_trivial
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try to find lower rank decomposition
    # Lower bound is 28
    for target_rank in range(28, 50):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with integers/half-integers
            # Use small integers to help with rounding
            np.random.seed(target_rank * 1000 + attempt)
            U_init = np.random.randint(-1, 2, (n*m, target_rank)).astype(float)
            V_init = np.random.randint(-1, 2, (m*p, target_rank)).astype(float)
            W_init = np.random.randint(-1, 2, (p*n, target_rank)).astype(float)
            
            # Sometimes random integers don't work well, try random floats
            if attempt % 2 == 1:
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(target_rank):
                                val += U_x[i*m+j, t] * V_x[j*p+k, t] * W_x[k*n+i, t]
                            error += (val - 1.0) ** 2
                return error
            
            # Try to minimize
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12})
                
                if result.fun < 1e-6:
                    # Good solution found, round to half-integers
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Check rounded solution
                    if check_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), target_rank):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        best_rank = target_rank
                        break  # Found good rank, try to go lower
            except Exception:
                pass
    
    return best_result
