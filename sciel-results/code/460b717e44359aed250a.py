"""Optimized tensor decomposition search for 2x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M  # 8
MP = M * P  # 24
PN = P * N  # 12

def _build_target():
    """Build the matrix multiplication tensor as a 3D array T[i, j, k] = 1."""
    return np.ones((N, M, P), dtype=np.float64)

def _compute_residual(U, V, W, T):
    """Compute residual[i,j,k] = sum_c U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c] - T[i,j,k]"""
    rank = U.shape[1]
    residual = np.zeros((N, M, P), dtype=np.float64)
    for c in range(rank):
        u_c = U[:, c].reshape(N, M)
        v_c = V[:, c].reshape(M, P)
        w_c = W[:, c].reshape(P, N)
        # For each (i,j,k): contribution is u_c[i,j] * v_c[j,k] * w_c[k,i]
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    residual[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    residual -= T
    return residual

def _compute_gradients(U, V, W, T):
    """Compute gradients of ||residual||^2 w.r.t. U, V, W."""
    rank = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # Precompute residual
    residual = _compute_residual(U, V, W, T)
    
    for c in range(rank):
        u_c = U[:, c].reshape(N, M)
        v_c = V[:, c].reshape(M, P)
        w_c = W[:, c].reshape(P, N)
        
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    r = residual[i, j, k]
                    # d/dU[i*M+j,c] = r * V[j*P+k,c] * W[k*N+i,c]
                    dU[i * M + j, c] += 2 * r * V[j * P + k, c] * W[k * N + i, c]
                    # d/dV[j*P+k,c] = r * U[i*M+j,c] * W[k*N+i,c]
                    dV[j * P + k, c] += 2 * r * U[i * M + j, c] * W[k * N + i, c]
                    # d/dW[k*N+i,c] = r * U[i*M+j,c] * V[j*P+k,c]
                    dW[k * N + i, c] += 2 * r * U[i * M + j, c] * V[j * P + k, c]
    
    return dU, dV, dW

def _matmul_tensor_check(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    residual = _compute_residual(U, V, W, T)
    return np.max(np.abs(residual)) < tol

def _rank_from_matrices(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return U.shape[1]

def _structured_init(r, seed=42):
    """Create structured initializations exploiting the all-ones tensor symmetry."""
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Block-diagonal style - each rank term handles a subset of (i,j,k)
    # For the all-ones tensor, we can use:
    # U[i*M+j, c] = a_{i,c}, V[j*P+k, c] = b_{j,c}, W[k*N+i, c] = c_{k,c}
    # where sum_c a_{i,c} * b_{j,c} * c_{k,c} = 1
    
    # Try: partition indices into groups and assign coefficients
    # Simple approach: use Kronecker-like structure
    
    U_try = np.zeros((NM, r))
    V_try = np.zeros((MP, r))
    W_try = np.zeros((PN, r))
    
    # Initialize with structured pattern:
    # For each rank term c, assign it to handle certain (i,j,k) combinations
    # Use a cyclic assignment
    
    for c in range(r):
        # Each term gets a "phase" offset
        phase = 2.0 * np.pi * c / r
        
        for i in range(N):
            for j in range(M):
                U_try[i * M + j, c] = 1.0 + 0.5 * np.cos(phase + i * 1.0 + j * 0.5)
        
        for j in range(M):
            for k in range(P):
                V_try[j * P + k, c] = 1.0 + 0.5 * np.cos(phase + j * 1.0 + k * 0.5)
        
        for k in range(P):
            for i in range(N):
                W_try[k * N + i, c] = 1.0 + 0.5 * np.cos(phase + k * 1.0 + i * 0.5)
    
    return U_try, V_try, W_try

def _random_init(r, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    U_try = rng.randn(NM, r) * 0.5
    V_try = rng.randn(MP, r) * 0.5
    W_try = rng.randn(PN, r) * 0.5
    return U_try, V_try, W_try

def _optimize(U, V, W, T, max_iters=2000, lr_init=0.01, deadline=None):
    """Optimize using gradient descent with adaptive learning rate."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = lr_init
    best_loss = np.inf
    patience = 0
    max_patience = 200
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute loss
        residual = _compute_residual(U, V, W, T)
        loss = np.sum(residual ** 2)
        
        if loss < 1e-12:
            break
        
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Reduce learning rate
            lr *= 0.5
            patience = 0
            if lr < 1e-8:
                break
        
        # Compute gradients
        dU, dV, dW = _compute_gradients(U, V, W, T)
        
        # Gradient descent step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Occasionally try larger steps if loss is decreasing
        if it % 100 == 0 and it > 0:
            lr *= 1.1  # Slight increase
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    T = _build_target()
    
    # Schoolbook fallback
    best_rank = N * M * P  # 48
    best_U = np.zeros((NM, best_rank))
    best_V = np.zeros((MP, best_rank))
    best_W = np.zeros((PN, best_rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                best_U[i * M + j, col] = 1
                best_V[j * P + k, col] = 1
                best_W[k * N + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(NM, MP, PN) = 24
    # Try ranks from 24 upward
    
    found = False
    for r in range(24, 48):
        if time.time() > deadline - 10:  # Keep 10s buffer
            break
        
        # Try multiple initialization strategies
        inits = [
            ('structured', _structured_init(r, seed=r)),
            ('random1', _random_init(r, seed=r)),
            ('random2', _random_init(r, seed=r + 1000)),
            ('random3', _random_init(r, seed=r + 2000)),
        ]
        
        for name, (U0, V0, W0) in inits:
            if time.time() > deadline - 5:
                break
            
            U_opt, V_opt, W_opt = _optimize(U0, V0, W0, T, max_iters=1500, lr_init=0.005, deadline=deadline - 5)
            
            if _matmul_tensor_check(U_opt, V_opt, W_opt, T, tol=1e-9):
                best_rank = r
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                break
        
        if found:
            break
    
    # Final polish: try to round to half-integers
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    # Check if rounding preserved validity
    if not _matmul_tensor_check(U_out, V_out, W_out, T, tol=1e-6):
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
