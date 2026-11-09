"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0 # Safety margin before 150s kill

def tensor_error(params, rank):
    """
    Compute the squared error of the tensor contraction.
    params: flattened array of U, V, W (each rank*9)
    rank: the rank
    """
    n, m, p = 3, 3, 3
    size = n * m
    
    U = params[:rank * size].reshape(rank, size)
    V = params[rank * size:2 * rank * size].reshape(rank, size)
    W = params[2 * rank * size:3 * rank * size].reshape(rank, size)
    
    # Transpose to make indexing easier for contraction
    # U is (rank, n*m), index c, i*m+j
    # V is (rank, m*p), index c, j*p+k
    # W is (rank, p*n), index c, k*n+i
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Sum over c
                # U[:, u_idx] * V[:, v_idx] * W[:, w_idx]
                prod = U[:, u_idx] * V[:, v_idx] * W[:, w_idx]
                val = np.sum(prod)
                error += (val - 1.0) ** 2
    return error

def recover_rationals(params, rank, tol=1e-6):
    """
    Attempt to recover integer/half-integer coefficients from floating point solution.
    """
    n, m, p = 3, 3, 3
    size = n * m
    
    U = params[:rank * size].reshape(rank, size)
    V = params[rank * size:2 * rank * size].reshape(rank, size)
    W = params[2 * rank * size:3 * rank * size].reshape(rank, size)
    
    U_rat = np.zeros((rank, size), dtype=object)
    V_rat = np.zeros((rank, size), dtype=object)
    W_rat = np.zeros((rank, size), dtype=object)
    
    max_denom = 1000
    
    for c in range(rank):
        for idx in range(size):
            # Try to find a simple fraction
            try:
                U_rat[c, idx] = Fraction(U[c, idx]).limit_denominator(max_denom)
                V_rat[c, idx] = Fraction(V[c, idx]).limit_denominator(max_denom)
                W_rat[c, idx] = Fraction(W[c, idx]).limit_denominator(max_denom)
            except:
                U_rat[c, idx] = 0
                V_rat[c, idx] = 0
                W_rat[c, idx] = 0
                
    # Verify the recovered rational solution exactly
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    val += U_rat[c, u_idx] * V_rat[c, v_idx] * W_rat[c, w_idx]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return U_rat, V_rat, W_rat
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Attempts to find a rank < 27 solution via optimization, falls back to 27.
    """
    n, m, p = 3, 3, 3
    size = n * m
    
    # Fallback: Schoolbook Rank 27
    U_fb = np.zeros((size, 27), dtype=float)
    V_fb = np.zeros((size, 27), dtype=float)
    W_fb = np.zeros((size, 27), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    # Verify fallback
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(27):
                    val += U_fb[u_idx, c] * V_fb[v_idx, c] * W_fb[w_idx, c]
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        return {"U": [], "V": [], "W": []}

    # Try to find a better solution (rank < 27)
    # We will try ranks 23, 24, 25, 26
    # Using a simple local optimization with random restarts
    
    best_rank = 27
    best_sol = None
    
    for target_rank in [23, 24, 25, 26]:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 30:
                break
                
            # Initialize with random integers in {-1, 0, 1}
            # Scale by 0.5 to allow half-integers easily?
            # Let's just use floats and snap later.
            
            U_init = np.random.randint(-1, 2, (size, target_rank)) * 0.5
            V_init = np.random.randint(-1, 2, (size, target_rank)) * 0.5
            W_init = np.random.randint(-1, 2, (size, target_rank)) * 0.5
            
            # Flatten
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective
            def objective(params):
                U = params[:target_rank * size].reshape(size, target_rank)
                V = params[target_rank * size:2 * target_rank * size].reshape(size, target_rank)
                W = params[2 * target_rank * size:3 * target_rank * size].reshape(size, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[:, u_idx] * V[:, v_idx], W[:, w_idx])
                            error += (val - 1.0) ** 2
                return error
                
            # Optimize
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    # Recover rationals
                    U_opt = res.x[:target_rank * size].reshape(size, target_rank)
                    V_opt = res.x[target_rank * size:2 * target_rank * size].reshape(size, target_rank)
                    W_opt = res.x[2 * target_rank * size:3 * target_rank * size].reshape(size, target_rank)
                    
                    # Snap to half-integers
                    U_rat = np.zeros((size, target_rank), dtype=object)
                    V_rat = np.zeros((size, target_rank), dtype=object)
                    W_rat = np.zeros((size, target_rank), dtype=object)
                    
                    for r in range(target_rank):
                        for idx in range(size):
                            U_rat[idx, r] = Fraction(U_opt[idx, r]).limit_denominator(100)
                            V_rat[idx, r] = Fraction(V_opt[idx, r]).limit_denominator(100)
                            W_rat[idx, r] = Fraction(W_opt[idx, r]).limit_denominator(100)
                            
                    # Verify
                    is_valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                for c in range(target_rank):
                                    val += U_rat[u_idx, c] * V_rat[v_idx, c] * W_rat[w_idx, c]
                                if val != 1:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                            
                    if is_valid:
                        best_rank = target_rank
                        best_sol = {"U": U_rat.tolist(), "V": V_rat.tolist(), "W": W_rat.tolist()}
                        break # Found a solution for this rank, break restarts
            except Exception:
                pass
                
        if best_sol is not None:
            break # Found a solution, stop trying lower ranks

    if best_sol is not None:
        return best_sol
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
