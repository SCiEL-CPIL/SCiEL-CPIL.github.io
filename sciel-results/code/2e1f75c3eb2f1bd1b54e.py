"""
Matrix multiplication 3x4x8 tensor decomposition.

Strategy:
1. Try structured Kronecker decompositions (factor m=4 as 2x2, p=8 as 4x2 or 2x4).
2. Polish with Adam optimization and snap to half-integers.
3. Fall back to schoolbook (rank 96).
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

def _snap_and_verify(U, V, W, T_true):
    for scale in [2, 1, 4]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if _verify(U_r, V_r, W_r, T_true, atol=1e-9):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    if _verify(U, V, W, T_true, atol=1e-8):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _adam_optimize(U, V, W, T_true, lr, max_iters, time_limit):
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

def _strassen_decomp(n, m, p):
    """
    Build a Kronecker-structured decomposition.
    Factor m = m1*m2, p = p1*p2.
    Decompose as: (n x m1 x p1) kron (m2 x p2) style.
    Actually for matrix mult tensor: T_{i,j,k} = delta.
    We can write j = j1*m2 + j2, k = k1*p2 + k2.
    Then the tensor is a Kronecker product of two smaller tensors.
    """
    pass

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = START_TIME + TIME_LIMIT - 10

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

    # --- Structured Kronecker approach ---
    # m=4=2*2, p=8=4*2 or 2*4 or 8*1
    # Try: treat j as (j1,j2) with j1 in {0,1}, j2 in {0,1,2,3} (m1=2,m2=4)
    #      and k as (k1,k2) with k1 in {0,1}, k2 in {0,1,2,3,4,5,6,7} (p1=2,p2=8)
    # The tensor T[i, j1*m2+j2, k1*p2+k2] = 1 if j1*m2+j2 is valid and k1*p2+k2 valid
    # Actually the tensor is: T[i, j, k] = 1 for the (i,j,k) entry.
    # Kronecker structure: if we factor j=(j1,j2), k=(k1,k2), the tensor is
    # T[i, j1*m2+j2, k1*p2+k2] which is NOT a Kronecker product in general.
    
    # Better: use the block structure. For each (j1,k1) pair, we have a
    # sub-tensor T_block[i, j2, k2] = 1. This is just the (n x m2 x p2) tensor.
    # So the full tensor is a block matrix where block (j1,k1) is the full
    # (n x m2 x p2) multiplication tensor.
    # If we decompose each block with rank r_block, total rank = (m1*p1) * r_block
    # but we can share structure across blocks.
    
    # Alternative: use the fact that the tensor is the outer product structure.
    # T[i,j,k] = 1. This is a rank-1 tensor in the sense that it's the product
    # of three indicator vectors, but the rank as a 3-tensor is n*m*p.
    
    # Let's try a direct approach: use Strassen-like for the 2x2 part.
    # Split j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}.
    # Split k = k1*4 + k2 where k1 in {0,1}, k2 in {0,1,2,3}.
    # For fixed (j1,k1), the sub-tensor is T[i, j1*2+j2, k1*4+k2] for i in {0,1,2},
    # j2 in {0,1}, k2 in {0,1,2,3}. This is the 3x2x4 multiplication tensor.
    # There are 4 such blocks (j1 in {0,1}, k1 in {0,1}).
    # If 3x2x4 has rank r, and we can combine the 4 blocks efficiently...
    
    # For now, try to find a lower-rank decomposition via optimization
    # starting from structured initializations.
    
    # Try various target ranks with structured + random inits
    for target_rank in [64, 56, 48, 40, 36, 32, 28, 24, 20, 16]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Build schoolbook for initialization
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
        
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(restart * 42 + target_rank)
            
            if restart == 0:
                # Structured init: select subset of schoolbook columns
                selected = rng.choice(best_rank, size=target_rank, replace=False)
                U = U_sb[:, selected].copy()
                V = V_sb[:, selected].copy()
                W = W_sb[:, selected].copy()
                U += rng.randn(nm, target_rank) * 0.05
                V += rng.randn(mp, target_rank) * 0.05
                W += rng.randn(pn, target_rank) * 0.05
            elif restart == 1:
                # Random init
                U = rng.randn(nm, target_rank) * 0.3
                V = rng.randn(mp, target_rank) * 0.3
                W = rng.randn(pn, target_rank) * 0.3
            else:
                # Scale schoolbook to target rank
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                U += rng.randn(nm, target_rank) * 0.1
                V += rng.randn(mp, target_rank) * 0.1
                W += rng.randn(pn, target_rank) * 0.1
            
            for lr in [0.01, 0.005, 0.001, 0.0005]:
                if time.time() > deadline:
                    break
                U, V, W, err = _adam_optimize(U, V, W, T_true, lr, 200, deadline)
                if err < 1e-8:
                    result = _snap_and_verify(U, V, W, T_true)
                    if result is not None:
                        U_r, V_r, W_r = result
                        if _verify(U_r, V_r, W_r, T_true, atol=1e-6):
                            return {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
