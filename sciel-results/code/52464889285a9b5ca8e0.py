import numpy as np
import time
from scipy.optimize import minimize
import random

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    # Build the list of (a, b, c) triples
    # a = i*m + j, b = j*p + k, c = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_constraints = len(triples)  # 42

    size_U = n * m  # 6
    size_V = m * p  # 21
    size_W = p * n  # 14

    def schoolbook():
        rank = n * m * p  # 42
        U = np.zeros((size_U, rank))
        V = np.zeros((size_V, rank))
        W = np.zeros((size_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb

    def compute_residual(params, R):
        """Compute residual sum of squares for given parameters and rank R."""
        U = params[0:size_U*R].reshape(size_U, R)
        V = params[size_U*R:size_U*R + size_V*R].reshape(size_V, R)
        W = params[size_U*R + size_V*R:].reshape(size_W, R)
        
        residuals = np.zeros(num_constraints)
        for idx, (a, b, c) in enumerate(triples):
            residuals[idx] = np.dot(U[a, :], np.multiply(V[b, :], W[c, :])) - 1.0
            
        return np.sum(residuals * residuals)

    def verify(U, V, W, tol=1e-8):
        """Verify the decomposition is exact."""
        for (a, b, c) in triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val - 1.0) > tol:
                return False
        return True

    def try_rank(R, max_restarts=15, max_iter=500):
        """Try to find a decomposition of rank R."""
        if R >= best_rank:
            return None
            
        total_vars = R * (size_U + size_V + size_W)
        best_res = np.inf
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            # Different initialization strategies
            if restart == 0:
                # Perturbed schoolbook-like: start with sparse structure
                params0 = np.zeros(total_vars)
                # Initialize with some random structure
                np.random.seed(restart * 1000)
                params0 = np.random.randn(total_vars) * (1.0 / np.sqrt(R))
            elif restart == 1:
                # Small random
                params0 = np.random.randn(total_vars) * 0.1
            elif restart == 2:
                # Even smaller
                params0 = np.random.randn(total_vars) * 0.05
            else:
                # Random with varying scale
                scale = random.uniform(0.01, 0.5)
                params0 = np.random.randn(total_vars) * scale
            
            # Try multiple optimizers
            methods = ['L-BFGS-B', 'SLSQP']
            if restart < 5:
                methods.append('Nelder-Mead')
            
            for method in methods:
                if time.time() - start_time > DEADLINE - 5:
                    return None
                
                try:
                    if method == 'L-BFGS-B':
                        res = minimize(
                            compute_residual, 
                            params0, 
                            args=(R,),
                            method='L-BFGS-B', 
                            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                    elif method == 'SLSQP':
                        res = minimize(
                            compute_residual, 
                            params0, 
                            args=(R,),
                            method='SLSQP', 
                            options={'maxiter': max_iter, 'ftol': 1e-14}
                        )
                    else:  # Nelder-Mead
                        res = minimize(
                            compute_residual, 
                            params0, 
                            args=(R,),
                            method='Nelder-Mead', 
                            options={'maxiter': max_iter * 2, 'xatol': 1e-12, 'fatol': 1e-14}
                        )
                except Exception:
                    continue
                    
                if res.fun < best_res:
                    best_res = res.fun
                    best_params = res.x.copy()
                    
                if best_res < 1e-12:
                    break
            
            if best_res < 1e-12:
                break
        
        if best_params is not None and best_res < 1e-10:
            U = best_params[0:size_U*R].reshape(size_U, R)
            V = best_params[size_U*R:size_U*R + size_V*R].reshape(size_V, R)
            W = best_params[size_U*R + size_V*R:].reshape(size_W, R)
            
            if verify(U, V, W):
                return U, V, W
        return None

    # Try ranks from 41 down to 21
    # Focus more effort on ranks close to 42 first, then decrease
    for R in range(rank_fb - 1, 20, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # More restarts for higher ranks (closer to known solution)
        if R >= 38:
            max_restarts = 25
            max_iter = 300
        elif R >= 30:
            max_restarts = 15
            max_iter = 200
        else:
            max_restarts = 8
            max_iter = 100
        
        result = try_rank(R, max_restarts=max_restarts, max_iter=max_iter)
        
        if result is not None:
            U_try, V_try, W_try = result
            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, R
            # Continue searching for even lower rank
        else:
            # If we failed at this rank, it might still be possible
            # But for efficiency, we continue to lower ranks
            pass

    # Final verification of best result
    if best_rank < rank_fb:
        if not verify(best_U, best_V, best_W, tol=1e-7):
            # Fallback to schoolbook
            best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
