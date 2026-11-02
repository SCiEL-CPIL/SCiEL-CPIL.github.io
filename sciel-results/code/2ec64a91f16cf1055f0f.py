"""Search for rank-minimal exact tensor decomposition of 2x3x3 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS

def _build_tensor():
    """Build the matrix multiplication tensor T[i, j, k] = 1 for all valid indices.
    The tensor is defined such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i, j, k
    """
    T = np.ones((n, m, p))
    return T

def _verify(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(result, 1.0, atol=tol)

def _objective(U, V, W):
    """Compute the error of the decomposition."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.sum((result - 1.0) ** 2)

def _gradients(U, V, W):
    """Compute gradients of the objective function."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    residual = result - 1.0
    
    # gU[i,j,c] = sum_k residual[i,j,k] * V[j,k,c] * W[k,i,c]
    gU = np.einsum('ijk,jkc,kic->ijc', residual, Vr, Wr)
    # gV[j,k,c] = sum_i residual[i,j,k] * U[i,j,c] * W[k,i,c]
    gV = np.einsum('ijk,ijc,kic->jkc', residual, Ur, Wr)
    # gW[k,i,c] = sum_j residual[i,j,k] * U[i,j,c] * V[j,k,c]
    gW = np.einsum('ijk,ijc,jkc->kic', residual, Ur, Vr)
    
    return gU.reshape(U.shape), gV.reshape(V.shape), gW.reshape(W.shape)

def _refine(U, V, W, max_iter=2000, lr_start=0.1, lr_end=0.001, tol=1e-14):
    """Refine with adaptive learning rate gradient descent (vectorized)."""
    current_obj = _objective(U, V, W)
    if current_obj < tol:
        return U, V, W
        
    for it in range(max_iter):
        progress = it / max_iter
        lr = lr_start * (lr_end / lr_start) ** progress
        
        gU, gV, gW = _gradients(U, V, W)
        
        U_new = U - lr * gU
        V_new = V - lr * gV
        W_new = W - lr * gW
        
        new_obj = _objective(U_new, V_new, W_new)
        
        if new_obj < current_obj:
            U, V, W = U_new, V_new, W_new
            current_obj = new_obj
        else:
            # Backtracking line search
            lr = lr * 0.5
            while lr > 1e-8:
                U_new = U - lr * gU
                V_new = V - lr * gV
                W_new = W - lr * gW
                new_obj = _objective(U_new, V_new, W_new)
                if new_obj < current_obj:
                    U, V, W = U_new, V_new, W_new
                    current_obj = new_obj
                    break
                lr = lr * 0.5
            else:
                break
                
        if current_obj < tol:
            break
            
    return U, V, W

def _snap_to_half_integer(M):
    """Round matrix to nearest half-integers."""
    return np.round(M * 2) / 2.0

def _discrete_polish(U, V, W, max_iters=100, time_limit=5.0):
    """Perform local search on the half-integer lattice."""
    U = _snap_to_half_integer(U)
    V = _snap_to_half_integer(V)
    W = _snap_to_half_integer(W)
    
    current_obj = _objective(U, V, W)
    if current_obj < 1e-10:
        return U, V, W, current_obj
        
    start_time = time.time()
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Try flipping entries in U
        for idx in np.ndindex(U.shape):
            old_val = U[idx]
            U[idx] = old_val + 0.5
            new_obj = _objective(U, V, W)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                U[idx] = old_val - 0.5
                new_obj = _objective(U, V, W)
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    U[idx] = old_val
                    
        # Try flipping entries in V
        for idx in np.ndindex(V.shape):
            old_val = V[idx]
            V[idx] = old_val + 0.5
            new_obj = _objective(U, V, W)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                V[idx] = old_val - 0.5
                new_obj = _objective(U, V, W)
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    V[idx] = old_val
                    
        # Try flipping entries in W
        for idx in np.ndindex(W.shape):
            old_val = W[idx]
            W[idx] = old_val + 0.5
            new_obj = _objective(U, V, W)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                W[idx] = old_val - 0.5
                new_obj = _objective(U, V, W)
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    W[idx] = old_val
                    
        if not improved:
            break
            
    return U, V, W, current_obj

def _try_rank(rank, deadline, best_result):
    """Try to find a valid decomposition at given rank."""
    if time.time() > deadline:
        return best_result
    
    best_obj = float('inf')
    best_UVW = None
    
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline - 2:
            break
            
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Refine
        U, V, W = _refine(U, V, W, max_iter=1000)
        
        obj = _objective(U, V, W)
        
        # Discrete polish
        U_p, V_p, W_p, obj_p = _discrete_polish(U, V, W, max_iters=50, time_limit=2.0)
        
        if obj_p < best_obj:
            best_obj = obj_p
            best_UVW = (U_p.copy(), V_p.copy(), W_p.copy())
            
        if best_obj < 1e-9:
            if _verify(best_UVW[0], best_UVW[1], best_UVW[2], tol=1e-9):
                return {'U': best_UVW[0].tolist(), 'V': best_UVW[1].tolist(), 'W': best_UVW[2].tolist()}
                
    return best_result

def _schoolbook():
    """Fallback schoolbook decomposition of rank 18."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    
    # Try ranks from low to high
    # Lower bound estimation: for 2x3x3, rank 12 is a reasonable target to explore.
    for rank in range(12, 18):
        if time.time() > deadline - 10:
            break
        result = _try_rank(rank, deadline, best_result)
        if result is not best_result:
            best_result = result
            return best_result
    
    return best_result
