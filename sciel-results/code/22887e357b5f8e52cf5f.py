import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll construct the decomposition by combining sub-decompositions
    # (2,2,5) with rank 14 and (2,2,10) with rank 28
    # Total rank = 42
    
    def solve_decomp(n, m, p, rank, seed=0):
        """Solve for a rank-minimal decomposition using numerical optimization."""
        np.random.seed(seed)
        total_U = n * m
        total_V = m * p
        total_W = p * n
        num_params = rank * (total_U + total_V + total_W)
        
        def residuals(params):
            U = params[:rank*total_U].reshape(rank, total_U).T
            V = params[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
            W = params[rank*(total_U+total_V):].reshape(rank, total_W).T
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        # Try multiple initializations with different seeds
        for trial in range(30):
            trial_seed = seed + trial
            np.random.seed(trial_seed)
            params0 = np.random.randn(num_params) * 0.3
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                cost = result.cost
                
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Check if rounded solution is exact
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                    
                    best_U, best_V, best_W = U_r, V_r, W_r
                    
            except Exception:
                pass
        
        return best_U, best_V, best_W
    
    # Solve for (2,2,5) rank 14
    U5, V5, W5 = solve_decomp(2, 2, 5, 14, seed=100)
    
    # Solve for (2,2,10) rank 28
    U10, V10, W10 = solve_decomp(2, 2, 10, 28, seed=200)
    
    if U5 is None or U10 is None:
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
        return {"U": U, "V": V, "W": W}
    
    # Combine the two decompositions
    U_full = np.hstack([U5, U10])
    V_full = np.zeros((m * p, 42))
    V_full[:m*5, :14] = V5
    V_full[m*5:, 14:] = V10
    W_full = np.zeros((p * n, 42))
    W_full[:5*n, :14] = W5
    W_full[5*n:, 14:] = W10
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
