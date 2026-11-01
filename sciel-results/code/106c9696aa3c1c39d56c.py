import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank 5 decomposition for (2, 3, 1)
    # Sub-problem: A (2x3), B (3x1), C (2x1)
    # U (2x3x5), V (3x1x5), W (2x1x5)
    # Condition: sum_r U[i',j',r] V[j,r] W[i,r] = delta[i',i] delta[j',j]
    
    R_sub = 5
    n_sub, m_sub, p_sub = 2, 3, 1
    
    # Flatten U, V, W for optimization
    # U: 2*3*R_sub = 30
    # V: 3*1*R_sub = 15
    # W: 2*1*R_sub = 10
    # Total 55 variables
    
    def residual(params):
        U = params[:30].reshape(2, 3, R_sub)
        V = params[30:45].reshape(3, 1, R_sub)
        W = params[45:55].reshape(2, 1, R_sub)
        
        res = []
        for i_prime in range(n_sub):
            for j_prime in range(m_sub):
                for i in range(n_sub):
                    for j in range(m_sub):
                        # Sum over r
                        val = 0
                        for r in range(R_sub):
                            val += U[i_prime, j_prime, r] * V[j, r] * W[i, r]
                        target = 1.0 if (i_prime == i and j_prime == j) else 0.0
                        res.append(val - target)
        return res
    
    # Initial guess
    np.random.seed(42)
    x0 = np.random.randn(55) * 0.1
    
    best_rank = n * m * p
    best_result = None
    
    # Try to solve
    try:
        sol = least_squares(residual, x0, method='lm', max_nfev=10000)
        if sol.success and sol.cost < 1e-10:
            U_sub = sol.x[:30].reshape(2, 3, R_sub)
            V_sub = sol.x[30:45].reshape(3, 1, R_sub)
            W_sub = sol.x[45:55].reshape(2, 1, R_sub)
            
            # Round to half-integers
            U_sub = np.round(U_sub * 2) / 2
            V_sub = np.round(V_sub * 2) / 2
            W_sub = np.round(W_sub * 2) / 2
            
            # Verify exactness
            res_check = residual(sol.x) # Use original for check? No, check rounded
            # Re-check with rounded values
            U_r = U_sub
            V_r = V_sub
            W_r = W_sub
            err = 0
            for i_prime in range(n_sub):
                for j_prime in range(m_sub):
                    for i in range(n_sub):
                        for j in range(m_sub):
                            val = 0
                            for r in range(R_sub):
                                val += U_r[i_prime, j_prime, r] * V_r[j, r] * W_r[i, r]
                            target = 1.0 if (i_prime == i and j_prime == j) else 0.0
                            err += abs(val - target)
            
            if err < 1e-9:
                # Construct full decomposition
                total_rank = p_sub * R_sub * (p // p_sub) # 9 * 5 = 45
                # Wait, p=9, p_sub=1. So 9 copies.
                total_rank = 45
                
                U = np.zeros((n*m, total_rank))
                V = np.zeros((m*p, total_rank))
                W = np.zeros((p*n, total_rank))
                
                for k in range(p):
                    for r in range(R_sub):
                        c = k * R_sub + r
                        for i in range(n):
                            for j in range(m):
                                U[i*m + j, c] = U_sub[i, j, r]
                        for j in range(m):
                            V[j*p + k, c] = V_sub[j, 0, r]
                        for i in range(n):
                            W[k*n + i, c] = W_sub[i, 0, r]
                
                best_rank = total_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    except Exception as e:
        pass
    
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_result
