"""Matrix multiplication 2x2x5 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 5]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 20


def _build_schoolbook():
    """Return (U, V, W) as numpy arrays for the schoolbook rank-20 decomposition."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _residual(U, V, W):
    """Compute the residual tensor: T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c].
    Returns max absolute value of residual."""
    # Build the multiplication tensor T[i,j,k] = 1 if the product C[i,k] = A[i,j]*B[j,k]
    # T is indexed as (i*m+j)*p+k or we use a 3D array
    # Actually the tensor for matrix mult: T[i,j,k] = 1 (it's the identity in the sense
    # that C[i,k] = sum_j A[i,j] B[j,k])
    # The tensor is T[i, j, k] = 1 for all i,j,k? No.
    # The matrix multiplication tensor has T[i,j,k] = 1 if and only if the product
    # involves A[i,j] and B[j,k] to produce C[i,k]. Actually T[i,j,k] = 1 for all valid
    # (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (the full tensor of ones? No.)
    # The matrix multiplication tensor: for C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Actually no - it's the structural tensor where T[i,j,k] = 1 iff the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C. This is always 1 for valid indices.
    # So T is just an all-ones tensor of shape (n, m, p).
    T = np.ones((n, m, p))
    
    # Compute sum_c U[:,c] * V[:,c] * W[:,c]
    # Reshape to 3D: result[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    result = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.max(np.abs(T - result))


def _residual_fast(U, V, W):
    """Faster residual computation using einsum."""
    T = np.ones((n, m, p))
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # result[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U3 = U.reshape(n, m, -1)  # (n, m, rank)
    V3 = V.reshape(m, p, -1)  # (m, p, rank)
    W3 = W.reshape(p, n, -1)  # (p, n, rank)
    
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.max(np.abs(T - result))


def _als_step(U, V, W, lr=0.1, max_iter=5000, tol=1e-10, deadline=None):
    """ALS optimization to minimize residual. Returns (U, V, W, residual)."""
    T = np.ones((n, m, p))
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    best_res = _residual_fast(U, V, W)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] = argmin sum_c (U[i*m+j,c]) * (V[j*p+k,c]*W[k*n+i,c])_k
        # The gradient wrt U[i*m+j, c] involves sum_k V[j*p+k, c] * W[k*n+i, c]
        # And the target is T[i,j,k] = 1 for all k
        # So we want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a linear system in U[i*m+j, :]:
        # For fixed (i,j), define A_kc = V[j*p+k, c] * W[k*n+i, c] for k in [0,p)
        # Then sum_c A_kc * U[i*m+j, c] = 1 for all k
        # This is A @ U[i*m+j, :] = 1 (vector of ones of length p)
        # Solve least squares
        
        for i in range(n):
            for j in range(m):
                # A is (p, rank): A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = (V3[j, :, :] * W3[:, i, :]).T  # (p, rank)
                b = np.ones(p)
                # Solve A @ x = b
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                U[i * m + j, :] = x
                U3[i, j, :] = x
        
        # Update V: for each (j,k), V[j*p+k, :] 
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        # A_ic = U[i*m+j, c] * W[k*n+i, c], want sum_c A_ic * V[j*p+k, c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = (U3[:, j, :] * W3[k, :, :]).T  # (n, rank)
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                V[j * p + k, :] = x
                V3[j, k, :] = x
        
        # Update W: for each (k,i), W[k*n+i, :]
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = (U3[i, :, :] * V3[:, k, :]).T  # (m, rank)
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                W[k * n + i, :] = x
                W3[k, i, :] = x
        
        res = _residual_fast(U, V, W)
        if res < best_res:
            best_res = res
        if res < tol:
            break
    
    return U, V, W, best_res


def _try_rank(r, deadline, max_restarts=30):
    """Try to find a rank-r decomposition. Returns (U, V, W, residual) or None."""
    if r >= RANK_SCHOOLBOOK:
        U, V, W = _build_schoolbook()
        return U, V, W, 0.0
    
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: start from schoolbook and zero out some columns, or random
        if restart == 0:
            # Start from schoolbook, keep first r columns
            U, V, W = _build_schoolbook()
            U = U[:, :r].copy()
            V = V[:, :r].copy()
            W = W[:, :r].copy()
        else:
            # Perturb schoolbook: keep r random columns
            U0, V0, W0 = _build_schoolbook()
            cols = random.sample(range(RANK_SCHOOLBOOK), r)
            U = U0[:, cols].copy()
            V = V0[:, cols].copy()
            W = W0[:, cols].copy()
            # Add small random perturbation
            U += 0.01 * np.random.randn(*U.shape)
            V += 0.01 * np.random.randn(*V.shape)
            W += 0.01 * np.random.randn(*W.shape)
        
        U, V, W, res = _als_step(U, V, W, deadline=deadline)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_res < 1e-10:
            break
    
    if best_res < 1e-8:
        return best_U, best_V, best_W, best_res
    return None


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try rank 17 first (known to be possible per knowledge)
    result = _try_rank(17, deadline, max_restarts=40)
    if result is not None:
        U, V, W, res = result
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 18
    result = _try_rank(18, deadline, max_restarts=40)
    if result is not None:
        U, V, W, res = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 19
    result = _try_rank(19, deadline, max_restarts=20)
    if result is not None:
        U, V, W, res = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 20
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
