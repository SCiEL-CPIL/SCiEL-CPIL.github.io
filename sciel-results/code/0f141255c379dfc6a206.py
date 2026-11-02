import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Number of constraints: n*m*p = 75
    # Variables: (n*m + m*p + p*n) * r = (15 + 25 + 15) * r = 55r
    
    def make_objective(r):
        """Create objective function for rank r"""
        def objective(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = x[n*m*r+m*p*r:].reshape(p*n, r)
            
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        errors.append(val - 1.0)
            return np.sum(np.array(errors)**2)
        return objective
    
    def make_jacobian(r):
        """Create Jacobian for rank r"""
        def jacobian(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = x[n*m*r+m*p*r:].reshape(p*n, r)
            
            jac = np.zeros((n*m*p, 55*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i*m*p + j*p + k
                        for c in range(r):
                            # d/dc U[i*m+j,c]
                            u_idx = (i*m+j)*r + c
                            jac[eq_idx, u_idx] = V[j*p+k, c] * W[k*n+i, c]
                            # d/dc V[j*p+k,c]
                            v_idx = n*m*r + (j*p+k)*r + c
                            jac[eq_idx, v_idx] = U[i*m+j, c] * W[k*n+i, c]
                            # d/dc W[k*n+i,c]
                            w_idx = (n*m+m*p)*r + (k*n+i)*r + c
                            jac[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
            return jac
        return jacobian
    
    best_rank = n*m*p  # schoolbook
    best_solution = None
    
    # Try ranks from low to high
    for r in range(2, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_trials = 5
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + r * 100 + trial)
            x0 = np.random.randn(55 * r) * 0.5
            
            obj_func = make_objective(r)
            jac_func = make_jacobian(r)
            
            try:
                result = minimize(obj_func, x0, jac=jac_func, method='BFGS',
                                  options={'maxiter': 500, 'gtol': 1e-10})
                if result.fun < 1e-10:
                    best_rank = r
                    best_solution = result.x
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            break
    
    # If we found a solution, convert to the required format
    if best_solution is not None:
        r = best_rank
        U = best_solution[:n*m*r].reshape(n*m, r)
        V = best_solution[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = best_solution[n*m*r+m*p*r:].reshape(p*n, r)
        
        # Convert to lists
        U_list = [[U[i, j] for j in range(r)] for i in range(n*m)]
        V_list = [[V[i, j] for j in range(r)] for i in range(m*p)]
        W_list = [[W[i, j] for j in range(r)] for i in range(p*n)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
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
