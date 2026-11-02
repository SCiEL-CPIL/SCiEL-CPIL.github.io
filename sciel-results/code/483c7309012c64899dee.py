"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
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
    
    # Check exactness
    def check_exact(U, V, W, n, m, p):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # QCQP approach: minimize sparsity penalty while maintaining exactness
    # Use penalty method: minimize ||residual||^2 + lambda * (number of non-zeros)
    
    def solve_qcqp(r, num_restarts=5):
        """Solve for rank-r decomposition using gradient-based optimization."""
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        for restart in range(num_restarts):
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Define objective: ||residual||^2 + lambda * L1 penalty
            lambda_val = 0.1
            
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Compute residual
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(r):
                                s += U_flat[i*m+j, t] * V_flat[j*p+k, t] * W_flat[k*n+i, t]
                            res += (s - 1.0) ** 2
                
                # L1 penalty for sparsity
                penalty = lambda_val * (np.abs(U_flat).sum() + np.abs(V_flat).sum() + np.abs(W_flat).sum())
                
                return res + penalty
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 100})
                
                if res.success:
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check if close to exact
                    exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for t in range(r):
                                    s += U_opt[i*m+j, t] * V_opt[j*p+k, t] * W_opt[k*n+i, t]
                                if abs(s - 1.0) > 1e-6:
                                    exact = False
                                    break
                        if not exact:
                            break
                    
                    if exact:
                        cost = res.fun
                        if cost < best_cost:
                            best_cost = cost
                            best_U, best_V, best_W = U_opt, V_opt, W_opt
            except:
                pass
        
        if best_U is not None:
            # Try to round to half-integers
            U_round = np.round(best_U * 2) / 2
            V_round = np.round(best_V * 2) / 2
            W_round = np.round(best_W * 2) / 2
            if check_exact(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        return None
    
    # Try ranks from lower bound upward
    for r in [30, 32, 35, 40, 45, 50, 55, 60]:
        result = solve_qcqp(r, num_restarts=3)
        if result:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
