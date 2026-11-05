"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=object)
    V = np.zeros((m * p, schoolbook_rank), dtype=object)
    W = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1

    def tensor_value(U_mat, V_mat, W_mat, rank):
        """Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    T[(i,j,k)] = s
        return T

    def verify(U_mat, V_mat, W_mat, rank):
        T = tensor_value(U_mat, V_mat, W_mat, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[(i,j,k)] != Fraction(1):
                        return False
        return True

    # Verify schoolbook
    if not verify(U, V, W, schoolbook_rank):
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}

    def try_rank32_numerical():
        """Use numerical optimization to find rank-32, then round and verify."""
        if time.time() > deadline - 30:
            return None
        
        rank = 32
        num_params = (n*m + m*p + p*n) * rank
        
        def objective(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
            W_flat = x[(n*m+m*p)*rank:].reshape(p*n, rank)
            
            errors = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        errors += (s - 1.0)**2
            
            return errors
        
        # Multiple random restarts
        np.random.seed(42)
        best_result = None
        best_fun = float('inf')
        
        for restart in range(5):
            if time.time() > deadline - 35:
                break
            
            x0 = np.random.randn(num_params) * 0.1
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-8})
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        
        if best_result is None or best_fun > 1e-4:
            return None
        
        U_sol = best_result.x[:n*m*rank].reshape(n*m, rank)
        V_sol = best_result.x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W_sol = best_result.x[(n*m+m*p)*rank:].reshape(p*n, rank)
        
        # Round to nearest half-integer
        def round_to_half(x):
            return Fraction(int(round(2*x)), 2)
        
        U_frac = np.zeros((n*m, rank), dtype=object)
        V_frac = np.zeros((m*p, rank), dtype=object)
        W_frac = np.zeros((p*n, rank), dtype=object)
        
        for i in range(n*m):
            for r in range(rank):
                U_frac[i, r] = round_to_half(U_sol[i, r])
        for i in range(m*p):
            for r in range(rank):
                V_frac[i, r] = round_to_half(V_sol[i, r])
        for i in range(p*n):
            for r in range(rank):
                W_frac[i, r] = round_to_half(W_sol[i, r])
        
        # Verify exactly
        if verify(U_frac, V_frac, W_frac, rank):
            return U_frac, V_frac, W_frac
        
        # Try local search: adjust individual entries by ±1/2
        improved = True
        while improved and time.time() < deadline - 40:
            improved = False
            for i in range(n*m):
                for r in range(rank):
                    old = U_frac[i, r]
                    for delta in [Fraction(1,2), Fraction(-1,2)]:
                        U_frac[i, r] = old + delta
                        if verify(U_frac, V_frac, W_frac, rank):
                            improved = True
                            break
                        U_frac[i, r] = old
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            for i in range(m*p):
                for r in range(rank):
                    old = V_frac[i, r]
                    for delta in [Fraction(1,2), Fraction(-1,2)]:
                        V_frac[i, r] = old + delta
                        if verify(U_frac, V_frac, W_frac, rank):
                            improved = True
                            break
                        V_frac[i, r] = old
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            for i in range(p*n):
                for r in range(rank):
                    old = W_frac[i, r]
                    for delta in [Fraction(1,2), Fraction(-1,2)]:
                        W_frac[i, r] = old + delta
                        if verify(U_frac, V_frac, W_frac, rank):
                            improved = True
                            break
                        W_frac[i, r] = old
                    if improved:
                        break
                if improved:
                    break
        
        if verify(U_frac, V_frac, W_frac, rank):
            return U_frac, V_frac, W_frac
        
        return None

    result = try_rank32_numerical()
    if result is not None:
        U_final, V_final, W_final = result
        return {"U": [[float(x) for x in row] for row in U_final], 
                "V": [[float(x) for x in row] for row in V_final], 
                "W": [[float(x) for x in row] for row in W_final]}

    # Return schoolbook
    return {"U": [[float(x) for x in row] for row in U], 
            "V": [[float(x) for x in row] for row in V], 
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
