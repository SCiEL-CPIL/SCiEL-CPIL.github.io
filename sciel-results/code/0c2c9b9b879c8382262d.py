import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = time.time() + 145.0  # 5s safety margin

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

def _verify_exact(n, m, p, U, V, W):
    """Verify exact decomposition. Entries should be int or half-int (multiples of 0.5)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    # Convert to Fraction for exact arithmetic
                    uc = Fraction(str(U[i * m + j][c]))
                    vc = Fraction(str(V[j * p + k][c]))
                    wc = Fraction(str(W[k * n + i][c]))
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True

def _try_als(n, m, p, target_rank, max_restarts=30, max_iters=200):
    """
    Alternating Least Squares to find a rank-r decomposition.
    Returns (U, V, W) as numpy arrays if found with low error, else None.
    """
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        if _remaining() < 5:
            break
        
        np.random.seed(42 + restart * 7)
        
        # Try different initialization strategies
        if restart % 3 == 0:
            # Small random
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
        elif restart % 3 == 1:
            # Structured: initialize with schoolbook-like pattern
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            # Fill first min(target_rank, n*m*p) columns with schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
            # Add small noise to break symmetry
            U += np.random.randn(n * m, target_rank) * 0.01
            V += np.random.randn(m * p, target_rank) * 0.01
            W += np.random.randn(p * n, target_rank) * 0.01
        else:
            # Random with larger scale
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        
        error = float('inf')
        for it in range(max_iters):
            if _remaining() < 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
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
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = z
                    except np.linalg.LinAlgError:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (s - 1.0) ** 2
            
            if error < 1e-12:
                break
        
        if error < best_error:
            best_error = error
            best_params = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-6 and best_params is not None:
        return best_params
    return None

def _try_round_and_verify(n, m, p, U, V, W, denominators=[1, 2, 4, 8]):
    """Try rounding to rationals with given denominators and verify exactly."""
    for denom in denominators:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        
        # Convert to list of lists of floats
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        
        if _verify_exact(n, m, p, U_list, V_list, W_list):
            return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    for target_rank in [49, 48, 47]:
        if _remaining() < 10:
            break
        
        params = _try_als(n, m, p, target_rank, max_restarts=40, max_iters=150)
        if params is not None:
            U, V, W = params
            rounded = _try_round_and_verify(n, m, p, U, V, W)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best_result
