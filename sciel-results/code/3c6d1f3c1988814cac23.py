import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import null_space
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = 2, 3, 6
    rank_schoolbook = n * m * p  # 36
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_solution = None
    best_rank = rank_schoolbook
    
    # Strategy: Search for structured low-rank decompositions
    # The tensor has structure: T is a sum of 36 rank-1 tensors
    # We can look for decompositions where U, V, W have specific patterns
    
    # Try ranks from 18 upward (half of schoolbook)
    # Use the fact that for 2x3x6, there might be rank ~18-24 solutions
    
    # Approach: Use variable elimination / structured search
    # For each rank R, set up the system and try to solve
    
    for R in range(18, 30):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations with structured constraints
        found = False
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            
            # Initialize with some structure
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Objective function with tighter tolerance
            def objective(params):
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R:].reshape(p*n, R)
                
                T_sum = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
                residual = T_sum - T
                return np.sum(residual**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_solution = (U_rounded, V_rounded, W_rounded, R)
                        found = True
                        break
                    
                    # If rounding fails, try integer rounding
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    T_check_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                    if np.allclose(T_check_int, T, atol=1e-9):
                        best_solution = (U_int, V_int, W_int, R)
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break
    
    # Also try a more structured approach: assume block structure
    # For 2x3x6, we can think of it as two 2x3x3 blocks
    
    if best_solution is None and time.time() < deadline - 20:
        # Try to find 2x3x3 decomposition and embed
        n2, m2, p2 = 2, 3, 3
        T233 = np.zeros((n2*m2, m2*p2, p2*n2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        for R233 in range(11, 16):
            if time.time() > deadline - 20:
                break
            
            for restart in range(15):
                if time.time() > deadline - 20:
                    break
                
                U233 = np.random.randn(n2*m2, R233) * 0.5
                V233 = np.random.randn(m2*p2, R233) * 0.5
                W233 = np.random.randn(p2*n2, R233) * 0.5
                
                def obj233(params):
                    U_f = params[:n2*m2*R233].reshape(n2*m2, R233)
                    V_f = params[n2*m2*R233:n2*m2*R233+m2*p2*R233].reshape(m2*p2, R233)
                    W_f = params[n2*m2*R233+m2*p2*R233:].reshape(p2*n2, R233)
                    T_sum = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
                    return np.sum((T_sum - T233)**2)
                
                x0 = np.concatenate([U233.flatten(), V233.flatten(), W233.flatten()])
                
                try:
                    res = minimize(obj233, x0, method='L-BFGS-B',
                                  options={'maxiter': 80, 'ftol': 1e-12})
                    
                    if res.fun < 1e-8:
                        U233_opt = res.x[:n2*m2*R233].reshape(n2*m2, R233)
                        V233_opt = res.x[n2*m2*R233:n2*m2*R233+m2*p2*R233].reshape(m2*p2, R233)
                        W233_opt = res.x[n2*m2*R233+m2*p2*R233:].reshape(p2*n2, R233)
                        
                        # Try half-integer rounding
                        U233_r = np.round(U233_opt * 2) / 2
                        V233_r = np.round(V233_opt * 2) / 2
                        W233_r = np.round(W233_opt * 2) / 2
                        
                        T_check = np.einsum('ir,jr,kr->ijk', U233_r, V233_r, W233_r)
                        if np.allclose(T_check, T233, atol=1e-9):
                            # Embed into 2x3x6
                            R236 = 2 * R233
                            U236 = np.zeros((n*m, R236))
                            V236 = np.zeros((m*p, R236))
                            W236 = np.zeros((p*n, R236))
                            
                            U236[:, :R233] = U233_r
                            V236[:m2*p2, :R233] = V233_r
                            W236[:p2*n2, :R233] = W233_r
                            
                            U236[:, R233:] = U233_r
                            V236[m2*p2:, R233:] = V233_r
                            W236[p2*n2:, R233:] = W233_r
                            
                            # Verify
                            T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
                            if np.allclose(T_final, T, atol=1e-9):
                                best_solution = (U236, V236, W236, R236)
                                break
                except Exception:
                    pass
            
            if best_solution is not None:
                break
    
    if best_solution:
        U, V, W, R = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
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
