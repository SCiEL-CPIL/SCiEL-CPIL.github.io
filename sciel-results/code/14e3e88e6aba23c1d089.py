"""
Matrix multiplication 3x4x8 tensor decomposition.

Strategy: Block-diagonal decomposition.
- Partition the 3 row indices into {0} and {1,2}.
- The 1x4x8 sub-tensor has rank 32 (schoolbook).
- The 2x4x8 sub-tensor: use structured decomposition.
  Split inner dim j=(j1,j2) and output k=(k1,k2) with j1,j2 in {0,1}, k1 in {0,1,2,3}, k2 in {0,1}.
  For each (j2,k2) pair, we have a 2x2x4 sub-problem. Use Strassen-like decomposition.
  
Actually, let's use a direct approach: build the schoolbook and verify, then try
to find improvements via targeted optimization with better initialization.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    """Verify that U, V, W correctly decompose the tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_and_verify(U, V, W, T_true):
    """Try to snap to half-integers and verify. Return snapped if valid."""
    for scale in [2, 1, 4]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if _verify(U_r, V_r, W_r, T_true, atol=1e-9):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    # Try without snapping
    if _verify(U, V, W, T_true, atol=1e-8):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _adam_optimize(U, V, W, T_true, lr, max_iters, time_limit):
    """Adam optimizer for tensor decomposition."""
    nm, r = U.shape
    mp, _ = V.shape
    pn, _ = W.shape
    
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        t = it + 1
        bc1 = 1 - b1**t
        bc2 = 1 - b2**t
        
        gU = np.einsum('abc,br,cr->ar', R, V, W)
        gV = np.einsum('abc,ar,cr->br', R, U, W)
        gW = np.einsum('abc,ar,br->cr', R, U, V)
        
        mU = b1*mU + (1-b1)*gU; vU = b2*vU + (1-b2)*gU**2
        mV = b1*mV + (1-b1)*gV; vV = b2*vV + (1-b2)*gV**2
        mW = b1*mW + (1-b1)*gW; vW = b2*vW + (1-b2)*gW**2
        
        mU_hat = mU/bc1; vU_hat = vU/bc2
        mV_hat = mV/bc1; vV_hat = vV/bc2
        mW_hat = mW/bc1; vW_hat = vW/bc2
        
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    return best_U, best_V, best_W, best_err

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    best_result = schoolbook()
    best_rank = n * m * p  # 96
    
    # Try to find lower-rank solutions using Adam optimization
    # Start from structured initializations
    deadline = START_TIME + TIME_LIMIT - 10
    
    # Strategy: Initialize from schoolbook subset + perturbation, then optimize
    for target_rank in [48, 40, 36, 32, 28, 24]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Build schoolbook
        U_sb = np.zeros((nm, best_rank))
        V_sb = np.zeros((mp, best_rank))
        W_sb = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        
        num_restarts = 2
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(restart * 42 + target_rank)
            
            # Select subset of schoolbook columns
            selected = rng.choice(best_rank, size=target_rank, replace=False)
            U = U_sb[:, selected].copy()
            V = V_sb[:, selected].copy()
            W = W_sb[:, selected].copy()
            
            # Perturb
            U += rng.randn(nm, target_rank) * 0.05
            V += rng.randn(mp, target_rank) * 0.05
            W += rng.randn(pn, target_rank) * 0.05
            
            # Adam optimization with decreasing learning rate
            for lr in [0.01, 0.005, 0.001, 0.0005]:
                if time.time() > deadline:
                    break
                U, V, W, err = _adam_optimize(U, V, W, T_true, lr, 300, deadline)
                if err < 1e-8:
                    result = _snap_and_verify(U, V, W, T_true)
                    if result is not None:
                        U_r, V_r, W_r = result
                        if _verify(U_r, V_r, W_r, T_true, atol=1e-6):
                            return {"U": U_r, "V": V_r, "W": W_r}
        
        # Also try pure random init
        for restart in range(2):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(10000 + restart * 100 + target_rank)
            U = rng.randn(nm, target_rank) * 0.3
            V = rng.randn(mp, target_rank) * 0.3
            W = rng.randn(pn, target_rank) * 0.3
            
            for lr in [0.01, 0.005, 0.001]:
                if time.time() > deadline:
                    break
                U, V, W, err = _adam_optimize(U, V, W, T_true, lr, 300, deadline)
                if err < 1e-8:
                    result = _snap_and_verify(U, V, W, T_true)
                    if result is not None:
                        U_r, V_r, W_r = result
                        if _verify(U_r, V_r, W_r, T_true, atol=1e-6):
                            return {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
