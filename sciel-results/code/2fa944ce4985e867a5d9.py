import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _solve_mm_tensor(n, m, p, rank, max_iters=500, seed=42, time_budget=10):
    """
    Attempt to find a rank decomposition for M_{n,m,p} using gradient descent.
    """
    start_time = time.time()
    T_true = _build_true_tensor(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    # Try a few random initializations
    for init_seed in range(5):
        if time.time() - start_time > time_budget:
            break
            
        rng = np.random.RandomState(seed + init_seed)
        
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5
        
        eta = 0.1
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
                
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            err = np.max(np.abs(R))
            
            if err < 1e-6:
                # Try rounding
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    return (U_r, V_r, W_r)
                
                if err < 1e-8:
                    return (U, V, W)
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
                
            # Decay learning rate
            if it % 50 == 0 and it > 0:
                eta *= 0.8
                
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Strategy: 
    # 1. Try to solve M_{3,4,4} with rank 16. If successful, construct M_{3,4,8} with rank 32.
    # 2. Try to solve M_{3,4,8} directly with rank 32, 36, 40, 48.
    # 3. Fallback to schoolbook (96).
    
    # Step 1: Solve M_{3,4,4}
    n1, m1, p1 = 3, 4, 4
    target_rank_1 = 16
    sol_1 = _solve_mm_tensor(n1, m1, p1, target_rank_1, time_budget=20)
    
    if sol_1 is not None:
        U1, V1, W1 = sol_1
        # Construct M_{3,4,8}
        # U is 12x32, V is 32x32, W is 24x32
        U_348 = np.hstack([U1, U1])
        
        V_348 = np.zeros((32, 32))
        # Rows 0-15 correspond to k=0-3
        V_348[0:16, 0:16] = V1
        # Rows 16-31 correspond to k=4-7
        V_348[16:32, 16:32] = V1
        
        W_348 = np.zeros((24, 32))
        # Rows 0-11 correspond to k=0-3
        W_348[0:12, 0:16] = W1
        # Rows 12-23 correspond to k=4-7
        W_348[12:24, 16:32] = W1
        
        # Verify
        if _verify_exact(U_348, V_348, W_348, n, m, p):
            return {
                "U": U_348.tolist(),
                "V": V_348.tolist(),
                "W": W_348.tolist()
            }

    # Step 2: Try direct solutions for M_{3,4,8}
    for rank in [32, 36, 40, 48]:
        sol = _solve_mm_tensor(n, m, p, rank, time_budget=20)
        if sol is not None:
            U, V, W = sol
            if _verify_exact(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    # Step 3: Fallback
    return _schoolbook(n, m, p)
