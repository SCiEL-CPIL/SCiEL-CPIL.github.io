import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    rank = len(U[0])
    
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if val != 1:
                    return False
    return True

def _try_polish(U, V, W, n, m, p, max_denom=100):
    """Try to snap to small rationals and verify."""
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for denom in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _gradient_descent(n, m, p, r, max_time=10, seed=42):
    """Gradient descent to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    for iteration in range(20000):
        if time.time() - start_time > max_time:
            break
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-12:
            break
        
        dL_dU_3d = 2 * np.einsum('ijk,jkc,kic->ijc', T - 1.0, V_3d, W_3d)
        dL_dV_3d = 2 * np.einsum('ijk,ijc,kic->jkc', T - 1.0, U_3d, W_3d)
        dL_dW_3d = 2 * np.einsum('ijk,ijc,jkc->kic', T - 1.0, U_3d, V_3d)
        
        gU = dL_dU_3d.reshape(n * m, r)
        gV = dL_dV_3d.reshape(m * p, r)
        gW = dL_dW_3d.reshape(p * n, r)
        
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        if iteration % 100 == 0:
            lr *= 0.99
    
    ok, final_err = _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-8)
    if ok or final_err < 1e-6:
        return best_U, best_V, best_W, r
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 20 to 30
    for r in range(20, 31):
        if time.time() > deadline - 10:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            res = _gradient_descent(n, m, p, r, max_time=2, seed=seed)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    polished = _try_polish(U, V, W, n, m, p)
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        # If polish fails, try harder polish or return floats?
                        # The task requires integer/half-integer entries.
                        # If we can't polish, we should not return floats.
                        # But if we found a valid float solution, maybe we can try more polishing.
                        pass
                    break
        
        if best_rank <= 22:
            break
    
    return result
