"""
Matrix multiplication 3x4x8 tensor decomposition.
Uses structured block decomposition with Strassen for 2x2x2 sub-blocks
and systematic search for further rank reduction.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W, T_true):
    for scale in [1, 2, 4]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if _verify(U_r, V_r, W_r, T_true, atol=1e-6):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return U.tolist(), V.tolist(), W.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10):
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix mult."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # C11 = A11*(B11-B22), C12 = A11*B12 + A12*B22, etc.
    # For tensor T[i,j,k] with i,j in {0,1}, k in {0,1}
    # U[i,j], V[j,k], W[k,i]
    U = np.array([
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1]
    ], dtype=np.float64)
    V = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ], dtype=np.float64)
    W = np.array([
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0]
    ], dtype=np.float64)
    return U, V, W

def _build_strassen_based_decomposition(n, m, p):
    """
    Build decomposition using Strassen for 2x2x2 blocks.
    Partition j into pairs: (0,1), (2,3)
    Partition k into pairs: (0,1), (2,3), (4,5), (6,7)
    For each (j_pair, k_pair), use Strassen if n=2, else schoolbook for n=3.
    
    For n=3, each 3x2x2 block has schoolbook rank 12.
    Total: 2 j_pairs * 4 k_pairs * 12 = 96 (same as schoolbook)
    
    But we can try to share terms across adjacent blocks.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p  # Start with schoolbook
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_improve_with_search(U_init, V_init, W_init, T_true, r_target, max_restarts=3, time_budget=15.0):
    """Try to find rank-r decomposition starting from initialization."""
    start_time = time.time()
    nm, mp, pn = U_init.shape[0], V_init.shape[0], W_init.shape[0]
    
    best_result = None
    best_err = np.inf
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Add small random perturbation
        rng = np.random.RandomState(restart * 1000 + 42)
        U += rng.randn(*U.shape) * 0.01
        V += rng.randn(*V.shape) * 0.01
        W += rng.randn(*W.shape) * 0.01
        
        eta = 0.1
        for phase in range(5):
            max_iters = max(100, 1000 // (phase + 1))
            U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
            
            if err < 1e-8:
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W, T_true)
                if _verify(U_s, V_s, W_s, T_true, atol=1e-6):
                    return U_s, V_s, W_s
                if _verify(U, V, W, T_true, atol=1e-6):
                    return U.tolist(), V.tolist(), W.tolist()
            
            eta *= 0.5
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
    
    if best_result is not None and best_err < 1e-6:
        U, V, W = best_result
        U_s, V_s, W_s = _snap_to_half_inters(U, V, W, T_true)
        if _verify(U_s, V_s, W_s, T_true, atol=1e-6):
            return U_s, V_s, W_s
        if _verify(U, V, W, T_true, atol=1e-6):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    # Start with schoolbook (rank 96)
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to improve using gradient-based search from schoolbook init
    U_init, V_init, W_init, _ = _build_strassen_based_decomposition(n, m, p)
    
    # Try ranks below schoolbook
    candidate_ranks = [95, 94, 93, 92, 90, 88, 86, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32]
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        if r >= best_rank:
            continue
        
        # Create initialization by zeroing out some columns
        U_r = U_init[:, :r]
        V_r = V_init[:, :r]
        W_r = W_init[:, :r]
        
        time_budget = min(10.0, (TIME_LIMIT - 10 - (time.time() - START_TIME)) * 0.7)
        if time_budget < 3:
            break
        
        result = _try_improve_with_search(U_r, V_r, W_r, T_true, r, max_restarts=2, time_budget=time_budget)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, T_true, atol=1e-6):
                return {"U": U, "V": V, "W": W}
    
    return best_result
