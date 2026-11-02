"""Matrix multiplication 3,3,4 rank decomposition."""
import time
from fractions import Fraction
import random
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify the decomposition exactly for all (i,j,k) triples."""
    n, m, p = DIMENSIONS
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_R(R, timeout=60.0, deadline=None):
    """Try to find a rank-R decomposition using random search and local optimization."""
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Flatten the tensor constraints into a vector of 36 equations
    # We want to find U, V, W such that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # This is a system of 36 polynomial equations in 3*R*9 + 3*R*12 + 3*R*12 variables? No, R*(9+12+12) variables.
    
    # Strategy: 
    # 1. Initialize U, V, W randomly with small integers or rationals.
    # 2. Use scipy.optimize.least_squares to minimize the residual of the tensor equations.
    # 3. If the residual is small, try to round to nearest half-integers and verify.
    
    def flatten_factors(U, V, W):
        return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
    
    def unflatten_factors(x):
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        U_flat = x[:U_size].reshape(n*m, R)
        V_flat = x[U_size:U_size+V_size].reshape(m*p, R)
        W_flat = x[U_size+V_size:].reshape(p*n, R)
        return U_flat, V_flat, W_flat

    def residuals(x):
        U_flat, V_flat, W_flat = unflatten_factors(x)
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    num_trials = 50
    
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > timeout:
            break
            
        # Random initialization
        # Use values in [-1, 1]
        x0 = np.random.uniform(-1, 1, size=(n*m*R + m*p*R + p*n*R))
        
        try:
            # Use least_squares to find a real solution
            # This is a non-linear least squares problem
            res = opt.least_squares(residuals, x0, method='lm', max_nfev=1000)
            
            if res.success or res.cost < 1e-6:
                U_flat, V_flat, W_flat = unflatten_factors(res.x)
                
                # Try to round to half-integers
                # The problem allows integer/half-integer entries
                # Let's try rounding each entry to the nearest 0.5
                U_rounded = np.round(U_flat * 2) / 2
                V_rounded = np.round(V_flat * 2) / 2
                W_rounded = np.round(W_flat * 2) / 2
                
                # Verify the rounded solution
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding fails, maybe the solution is already close to integers?
                # Try rounding to integers
                U_int = np.round(U_flat)
                V_int = np.round(V_flat)
                W_int = np.round(W_flat)
                
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            pass

    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin
    
    # Start with schoolbook (rank 36)
    best = _build_schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find lower rank decompositions
    # We know rank 36 is possible.
    # Let's try ranks 30, 31, 32, 33, 34, 35.
    # A known upper bound for 3x3x4 is 30 (Strassen-like improvements).
    
    for R in range(30, best_rank):
        if time.time() > deadline - 5.0:
            break
        result = _try_rank_R(R, timeout=20.0, deadline=deadline)
        if result is not None:
            best = result
            best_rank = R
            break # Found a lower rank, stop searching for even lower? 
                   # Actually, we want the MINIMAL rank. So we should keep going down.
                   # But the loop goes UP from 30. So we should go DOWN from best_rank-1.
    
    # Let's restart the loop logic: try from best_rank-1 down to 30.
    # But finding rank 30 is hard. Maybe we should try 35, 34, 33, 32, 31, 30.
    # The previous loop went 30 to 35. If it finds 35, it returns.
    # If we want the minimal rank, we should try the lowest possible rank first?
    # Or highest possible improvement first?
    # Usually, it's easier to find a rank slightly lower than schoolbook than a rank much lower.
    # So let's try 35, 34, 33, 32, 31, 30.
    
    # Reset best to schoolbook
    best = _build_schoolbook()
    best_rank = len(best["U"][0])
    
    for R in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 5.0:
            break
        result = _try_rank_R(R, timeout=15.0, deadline=deadline)
        if result is not None:
            best = result
            best_rank = R
    
    return best
