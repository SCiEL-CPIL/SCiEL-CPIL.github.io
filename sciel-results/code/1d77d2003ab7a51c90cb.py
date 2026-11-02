import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.allclose(T_rec, T, atol=tol)

def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded))
    return rounded, max_err

def _als_decompose(T, n, m, p, rank, max_iters=200, tol=1e-10, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    """
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
    else:
        U = np.random.randn(n * m, rank) * 0.5
    
    if init_V is not None:
        V = np.array(init_V, dtype=float).copy()
    else:
        V = np.random.randn(m * p, rank) * 0.5
    
    if init_W is not None:
        W = np.array(init_W, dtype=float).copy()
    else:
        W = np.random.randn(p * n, rank) * 0.5
    
    prev_loss = np.inf
    for _ in range(max_iters):
        # Solve for U
        TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_U = np.einsum('abc,bcd->ad', T, TW)
        G_U = np.einsum('bcd,bce->de', TW, TW)
        try:
            U_new = np.linalg.solve(G_U + 1e-10 * np.eye(rank), h_U.T).T
        except np.linalg.LinAlgError:
            U_new = h_U
        U = U_new
        
        # Solve for V
        UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_V = np.einsum('abc,acd->bd', T, UW)
        G_V = np.einsum('acd,ace->de', UW, UW)
        try:
            V_new = np.linalg.solve(G_V + 1e-10 * np.eye(rank), h_V.T).T
        except np.linalg.LinAlgError:
            V_new = h_V
        V = V_new
        
        # Solve for W
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        h_W = np.einsum('abc,abd->cd', T, UV)
        G_W = np.einsum('abd,abe->de', UV, UV)
        try:
            W_new = np.linalg.solve(G_W + 1e-10 * np.eye(rank), h_W.T).T
        except np.linalg.LinAlgError:
            W_new = h_W
        W = W_new
        
        # Calculate loss
        T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
        loss = np.sum((T_rec - T) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U, V, W, loss

def _try_rank(T, n, m, p, target_rank, deadline, max_restarts=3, max_iters=80):
    """Try to find a valid decomposition of the given rank."""
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(1000 + restart * 100 + target_rank)
        
        try:
            U, V, W, loss = _als_decompose(T, n, m, p, target_rank, max_iters=max_iters, tol=1e-10)
            
            if loss < 1e-6:
                U_rounded, err_u = _round_to_halfint(U)
                V_rounded, err_v = _round_to_halfint(V)
                W_rounded, err_w = _round_to_halfint(W)
                
                max_err = max(err_u, err_v, err_w)
                
                if max_err < 1e-3:
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }, target_rank
        except Exception:
            continue
    
    return None, target_rank

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    T = _build_target_tensor(n, m, p)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Try ranks from low to high
    # Theoretical lower bound for 3x3x5 is at least 15 (since 3x3x3 has rank 7, and we need 2 more columns)
    # Realistic minimum might be around 17-20
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
            
        result, rank = _try_rank(T, n, m, p, target_rank, deadline, max_restarts=5, max_iters=100)
        
        if result is not None:
            best_rank = target_rank
            best_result = result
            break  # Found the minimum rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
