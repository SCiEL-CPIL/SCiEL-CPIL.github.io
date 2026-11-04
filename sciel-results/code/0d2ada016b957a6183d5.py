"""Rewritten program: search for a lower-rank exact decomposition of the 3x5 @ 5x6 tensor."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 135.0  # 15s safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    """Return the trivial rank 90 decomposition."""
    rank = NM * MP // MP * 1  # just n*m*p = 90
    rank = n * m * p
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matmul tensor.
    Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    For exact decomposition, T should equal 1 for all (i,j,k).
    """
    rank = U.shape[1]
    # Build the tensor via einsum
    # U: (NM, rank) -> reshape to (n, m, rank)
    # V: (MP, rank) -> reshape to (m, p, rank)
    # W: (PN, rank) -> reshape to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _random_init(rank, seed=None):
    """Random initialization with small integers/half-integers."""
    rng = np.random.RandomState(seed)
    # Use values from {-2, -1, -0.5, 0, 0.5, 1, 2}
    vals = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    U = rng.choice(vals, size=(NM, rank))
    V = rng.choice(vals, size=(MP, rank))
    W = rng.choice(vals, size=(PN, rank))
    return U, V, W

def _gradient_step(U, V, W, lr=0.01, seed=None):
    """One step of gradient descent on the loss ||T - 1||^2.
    T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    dL/dU[i,j,c] = 2 * sum_k (T[i,j,k]-1) * V[j,k,c] * W[k,i,c]
    Similarly for V and W.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    E = T - 1.0  # error
    
    # Gradients
    dU3 = 2 * np.einsum('ijk,jkc,kic->ijc', E, V3, W3)
    dV3 = 2 * np.einsum('ijk,ijc,kic->jkc', E, U3, W3)
    dW3 = 2 * np.einsum('ijk,ijc,jkc->kic', E, U3, V3)
    
    U_new = U3 - lr * dU3
    V_new = V3 - lr * dV3
    W_new = W3 - lr * dW3
    
    return U_new.reshape(NM, rank), V_new.reshape(MP, rank), W_new.reshape(PN, rank)

def _project_to_half_int(U, V, W):
    """Project entries to nearest half-integer."""
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U, V, W

def _search_rank(rank, max_iters=2000, seed=None):
    """Try to find a decomposition of given rank via gradient descent."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    U, V, W = _random_init(rank, seed=seed)
    lr = 0.05
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        U, V, W = _gradient_step(U, V, W, lr=lr)
        
        # Check convergence
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < 1e-8:
            # Project to half-integers and verify
            U_p, V_p, W_p = _project_to_half_int(U, V, W)
            if _verify(U_p, V_p, W_p, tol=1e-9):
                return U_p, V_p, W_p
        
        # Decay learning rate
        if it % 200 == 199:
            lr *= 0.8
    
    return None

def construct():
    """Search for a rank-minimal exact tensor decomposition."""
    # First, try to find a decomposition with rank < 90
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But for matmul, the rank is at least max(n*m, m*p, p*n) and at most n*m*p
    
    # Try ranks from lower bound up
    lower_bound = max(NM, MP, PN)  # 30
    upper_bound = n * m * p  # 90
    
    best_result = None
    
    # Try a few ranks below 90
    for rank in range(lower_bound, 90):
        if _time_left() < 10.0:
            break
        result = _search_rank(rank, max_iters=500)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _schoolbook()
