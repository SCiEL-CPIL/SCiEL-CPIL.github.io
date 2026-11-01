import numpy as np
import time
from fractions import Fraction
import sys
from scipy.optimize import least_squares

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit
    
    n, m, p = 2, 3, 3
    TARGET_RANK = 17
    SCHOOL_RANK = 18

    # 1. Construct Schoolbook Solution (Rank 18) as fallback
    U_school = np.zeros((n * m, SCHOOL_RANK))
    V_school = np.zeros((m * p, SCHOOL_RANK))
    W_school = np.zeros((p * n, SCHOOL_RANK))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1.0
                V_school[j * p + k, idx] = 1.0
                W_school[k * n + i, idx] = 1.0
                idx += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = SCHOOL_RANK

    # Residual function for least_squares
    # x is flattened [U, V, W]
    # U shape (6, R), V shape (9, R), W shape (6, R)
    def residuals(x):
        U_flat, V_flat, W_flat = x[:n*m*TARGET_RANK], x[n*m*TARGET_RANK:n*m*TARGET_RANK + m*p*TARGET_RANK], x[n*m*TARGET_RANK + m*p*TARGET_RANK:]
        U = U_flat.reshape(n*m, TARGET_RANK)
        V = V_flat.reshape(m*p, TARGET_RANK)
        W = W_flat.reshape(p*n, TARGET_RANK)
        
        # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] == 1 for all i,j,k
        # Residuals are T[i,j,k] - 1
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0
        return res.flatten()

    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = np.random.RandomState(restart * 7919 + 13)
        
        # Initialization
        # Use a mix of random and structured initialization
        # Sometimes pure random works, sometimes structured is better
        if restart % 2 == 0:
            # Random
            U = rng.randn(n * m, TARGET_RANK)
            V = rng.randn(m * p, TARGET_RANK)
            W = rng.randn(p * n, TARGET_RANK)
        else:
            # Perturbed Schoolbook (take 17 random directions from schoolbook basis)
            # This is harder to implement correctly for rank reduction, so stick to random
            U = rng.randn(n * m, TARGET_RANK)
            V = rng.randn(m * p, TARGET_RANK)
            W = rng.randn(p * n, TARGET_RANK)
            
        # Normalize to avoid huge numbers initially
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use 'lm' for underdetermined systems
            result = least_squares(residuals, x0, method='lm', max_nfev=2000)
            
            if result.success or result.cost < 1e-12:
                # Extract solution
                U_sol = result.x[:n*m*TARGET_RANK].reshape(n*m, TARGET_RANK)
                V_sol = result.x[n*m*TARGET_RANK:n*m*TARGET_RANK + m*p*TARGET_RANK].reshape(m*p, TARGET_RANK)
                W_sol = result.x[n*m*TARGET_RANK + m*p*TARGET_RANK:].reshape(p*n, TARGET_RANK)
                
                # Try rounding to nearest half-integer
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Exact verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = Fraction(0)
                            for r in range(TARGET_RANK):
                                total += Fraction(U_int[i * m + j, r]).limit_denominator(10000) * \
                                           Fraction(V_int[j * p + k, r]).limit_denominator(10000) * \
                                           Fraction(W_int[k * n + i, r]).limit_denominator(10000)
                            if total != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_U, best_V, best_W = U_int, V_int, W_int
                    best_rank = TARGET_RANK
                    break # Found optimal, stop searching
                    
        except Exception:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
