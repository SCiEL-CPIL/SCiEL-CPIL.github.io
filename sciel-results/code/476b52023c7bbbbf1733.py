import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_als(n, m, p, target_rank, max_restarts=100, time_budget=30.0):
    """
    Attempt to find a decomposition of the given rank using Alternating Least Squares (ALS).
    """
    start_time = time.time()
    if time.time() > _DEADLINE - 5:
        return None

    best_error = float('inf')
    best_params = None

    # Pre-compute indices for speed
    # U indices: (i, j) -> i*m+j
    # V indices: (j, k) -> j*p+k
    # W indices: (k, i) -> k*n+i

    for restart in range(max_restarts):
        if time.time() > start_time + time_budget or time.time() > _DEADLINE - 2:
            break
        
        np.random.seed(1000 + restart)
        
        # Initialization: Schoolbook-like
        # We have n*m*p = 50 terms. We want target_rank (e.g., 49).
        # Initialize U, V, W with small random noise, or copy schoolbook and drop one?
        # Dropping one and perturbing is a good start.
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Fill with schoolbook pattern for first target_rank terms
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c < target_rank:
                        U[i * m + j, c] = 1.0
                        V[j * p + k, c] = 1.0
                        W[k * n + i, c] = 1.0
                        c += 1
        
        # Add noise to help escape local minima
        noise_scale = 0.1
        U += np.random.normal(0, noise_scale, U.shape)
        V += np.random.normal(0, noise_scale, V.shape)
        W += np.random.normal(0, noise_scale, W.shape)
        
        # ALS Iterations
        for it in range(50): # Max ALS iterations per restart
            if time.time() > start_time + time_budget or time.time() > _DEADLINE - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # System: sum_c U_ij,c * (V_jk,c * W_ki,c) = 1 for all k
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A @ U_ij = ones(p)
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        # Use least squares
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    # System: sum_c V_jk,c * (U_ij,c * W_ki,c) = 1 for all i
                    # B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Solve B @ V_jk = ones(n)
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        y, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = y
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # System: sum_c W_ki,c * (U_ij,c * V_jk,c) = 1 for all j
                    # C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Solve C @ W_ki = ones(m)
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = z
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (s - 1.0)**2
            
            if error < 1e-8:
                break

        if error < best_error:
            best_error = error
            best_params = (U.copy(), V.copy(), W.copy())
            
            if best_error < 1e-6:
                break

    if best_error < 1e-4 and best_params is not None:
        U, V, W = best_params
        
        # Try rounding to rationals
        # The task requires integer/half-integer entries.
        # So we check denom=1 and denom=2.
        
        for denom in [1, 2, 4]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            # Convert to lists for verification
            # Note: _verify uses exact arithmetic on the inputs.
            # If we pass floats, it might have precision issues.
            # But if we rounded to 1/2 or 1/4, they are exact in binary.
            # However, _verify sums them up.
            # To be safe, convert to Fractions for verification?
            # The task says "integer/half-integer entries".
            # So denom 1 and 2 are the primary candidates.
            
            if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # If rounding to 1/2 failed, maybe it's integers?
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify(n, m, p, U_int.tolist(), V_int.tolist(), W_int.tolist()):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    # Budget: 60 seconds
    result = _try_rank_als(n, m, p, 49, max_restarts=200, time_budget=60.0)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
            
    # Try to find rank 46 (if time remains)
    if time.time() < _DEADLINE - 10:
        result_46 = _try_rank_als(n, m, p, 46, max_restarts=50, time_budget=20.0)
        if result_46 is not None:
            if _verify(n, m, p, result_46['U'], result_46['V'], result_46['W']):
                return result_46

    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
