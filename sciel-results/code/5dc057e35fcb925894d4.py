import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin before 150s kill

    # Schoolbook decomposition as guaranteed fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Verify schoolbook is correct
    U3 = np.array(U_sb).reshape(n, m, rank_sb)
    V3 = np.array(V_sb).reshape(m, p, rank_sb)
    W3 = np.array(W_sb).reshape(p, n, rank_sb)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if np.max(np.abs(T - 1.0)) < 1e-9:
        pass

    # The bilinear complexity of 3x3x7 multiplication:
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Upper bound: 63 (schoolbook)
    # We'll try to find lower rank decompositions using optimization

    def _try_rank(R, n_restarts=5):
        """Try to find a rank-R decomposition using multiple random restarts."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(1000 + R * 100 + restart)
            
            # Initialize with small random values
            U0 = np.random.randn(n * m, R) * 0.3
            V0 = np.random.randn(m * p, R) * 0.3
            W0 = np.random.randn(p * n, R) * 0.3
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n, m, R)
                V = x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
                W = x[n*m*R+m*p*R:].reshape(p, n, R)
                T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                residual = T - 1.0
                return np.sum(residual**2)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-9,
                                           'maxfun': 20000})
            except:
                continue
            
            if time.time() > deadline:
                return None
            
            if result.fun < 1e-8:
                # Found a valid decomposition
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Try to round to integers or half-integers
                # First try rounding to integers
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                
                U3_r = U_int.reshape(n, m, R)
                V3_r = V_int.reshape(m, p, R)
                W3_r = W_int.reshape(p, n, R)
                T_r = np.einsum('ijr,jkr,kir->ijk', U3_r, V3_r, W3_r)
                
                if np.max(np.abs(T_r - 1.0)) < 1e-6:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rounding to half-integers
                U_half = np.round(U_opt * 2) / 2
                V_half = np.round(V_opt * 2) / 2
                W_half = np.round(W_opt * 2) / 2
                
                U3_r = U_half.reshape(n, m, R)
                V3_r = V_half.reshape(m, p, R)
                W3_r = W_half.reshape(p, n, R)
                T_r = np.einsum('ijr,jkr,kir->ijk', U3_r, V3_r, W3_r)
                
                if np.max(np.abs(T_r - 1.0)) < 1e-6:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
                # Try more aggressive scaling - maybe the solution needs scaling
                # Try rounding to quarters
                U_quarter = np.round(U_opt * 4) / 4
                V_quarter = np.round(V_opt * 4) / 4
                W_quarter = np.round(W_opt * 4) / 4
                
                U3_r = U_quarter.reshape(n, m, R)
                V3_r = V_quarter.reshape(m, p, R)
                W3_r = W_quarter.reshape(p, n, R)
                T_r = np.einsum('ijr,jkr,kir->ijk', U3_r, V3_r, W3_r)
                
                if np.max(np.abs(T_r - 1.0)) < 1e-6:
                    return {"U": U_quarter.tolist(), "V": V_quarter.tolist(), "W": W_quarter.tolist()}
        
        return None

    # Try ranks from lower bound upward
    for target_rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 42, 45, 48, 50, 52, 54, 56, 58, 60, 62]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Use more restarts for smaller ranks
        n_restarts = 8 if target_rank <= 25 else 5 if target_rank <= 35 else 3
        
        result = _try_rank(target_rank, n_restarts)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
    
    return best_result
