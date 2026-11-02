"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import fsolve

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def make_equations(r):
        """Create the system of equations for given rank r."""
        def equations(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            eqs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        eqs.append(val - 1)
            return eqs
        return equations
    
    # Try to find lower rank decompositions
    for r in range(80, rank_school):  # Start from 80, go up to schoolbook
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        if n_vars < n*m*p:  # Need at least as many variables as equations
            continue
        
        # Try multiple random initializations
        for init in range(10):
            if time.time() > deadline:
                break
            
            x0 = np.random.randn(n_vars)
            
            def equations(x):
                U = x[:r*n*m].reshape(n*m, r)
                V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                W = x[r*n*m + r*m*p:].reshape(p*n, r)
                
                eqs = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            eqs.append(val - 1)
                return eqs
            
            try:
                sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=10000)
                if ier == 1:
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_sol[i*m+j, :] * V_sol[j*p+k, :] * W_sol[k*n+i, :])
                                max_err = max(max_err, abs(val - 1))
                    
                    if max_err < 1e-8:
                        best_rank = r
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        break
            except Exception:
                pass
        
        if best_rank == r:
            break
    
    # Convert to half-integers if possible
    def to_half_integer(arr):
        return np.round(arr * 2) / 2
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify rounded
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_out[i*m+j, :] * V_out[j*p+k, :] * W_out[k*n+i, :])
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-6:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
