"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U @ V^T @ W^T reconstructs the matmul tensor."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape into tensors
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    # T[i,j,k] = sum_c Ut[i,j,c] * Vt[j,k,c] * Wt[k,i,c]
    # Compute via einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    # Expected: T[i,j,k] = 1 for all i,j,k (since it's the identity map for matmul)
    # Actually, the matmul tensor has T[i,j,k] = 1 for all i,j,k
    # Because (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is all ones
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _rank(U, V, W):
    """Return the number of columns (rank) of the decomposition."""
    return U.shape[1]


def _try_rank(r, n, m, p, max_iter=500, seed=0, time_limit=None):
    """Try to find a decomposition of rank r using iterative methods."""
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_limit if time_limit else None
    
    # Initialize randomly
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    # Normalize initial columns
    for c in range(r):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 0:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        
        # Compute current tensor
        Ut = U.reshape(n, m, r)
        Vt = V.reshape(m, p, r)
        Wt = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        
        # Target tensor is all ones
        target = np.ones((n, m, p))
        err = np.sum((T - target)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Update U: for each (i,j), solve for U[i,j,:] given V and W
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Let M_U[i,j,k] = sum_c V[j,k,c] * W[k,i,c]  (shape m x p... no)
        # Actually: T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        # So for fixed i,j: T[i,j,:] = U[i,j,:] @ diag(V[j,:,c]*W[:,i,c]) summed over c
        # Let's define: for each c, the contribution to T[i,j,k] is U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Group by c: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For fixed i,j: vector T[i,j,:] (length p) = sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c])
        # Let B_c[j,k] = V[j,k,c] * W[k,i,c]  -- but this depends on i too
        # This is getting complex. Let's use a simpler gradient-based approach.
        
        # Gradient-based update
        # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] for each k
        # dT/dV[j,k,c] = U[i,j,c] * W[k,i,c] for each i
        # dT/dW[k,i,c] = U[i,j,c] * V[j,k,c] for each j
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # dU[i*m+j, c] = sum_k (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    # sum over k
                    dU[i*m+j, c] = np.sum((T[i, j, :] - target[i, j, :]) * V[j*p:(j+1)*p, c] * W[0:p, c] * np.ones(1) )
                    # Wait, W[k*n+i, c] for k in range(p)
                    # W indices: k*n+i for k=0..p-1
                    W_cols = W[np.array([k*n+i for k in range(p)]), c]
                    V_cols = V[j*p:(j+1)*p, c]
                    dU[i*m+j, c] = np.sum((T[i, j, :] - target[i, j, :]) * V_cols * W_cols)
        
        # dV[j*p+k, c] = sum_i (T[i,j,k] - 1) * U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    U_cols = U[np.array([i*m+j for i in range(n)]), c]
                    W_cols = W[np.array([k*n+i for i in range(n)]), c]
                    dV[j*p+k, c] = np.sum((T[:, j, k] - target[:, j, k]) * U_cols * W_cols)
        
        # dW[k*n+i, c] = sum_j (T[i,j,k] - 1) * U[i*m+j, c] * V[j*p+k, c]
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    U_cols = U[np.array([i*m+j for j in range(m)]), c]
                    V_cols = V[np.array([j*p+k for j in range(m)]), c]
                    dW[k*n+i, c] = np.sum((T[i, :, k] - target[i, :, k]) * U_cols * V_cols)
        
        # Gradient descent with adaptive step
        lr = 0.01 / (1 + it * 0.001)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Normalize columns to prevent blowup
        for c in range(r):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-10:
                scale = 1.0 / norm
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
    
    # Final check
    Ut = best_U.reshape(n, m, r)
    Vt = best_V.reshape(m, p, r)
    Wt = best_W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    target = np.ones((n, m, p))
    if np.allclose(T, target, atol=1e-5):
        return best_U, best_V, best_W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = _rank(U, V, W)
    best_result = (U, V, W)
    
    # Try to reduce rank
    for r in range(best_rank - 1, max(10, best_rank // 2), -1):
        if time.time() > deadline - 30:
            break
        
        time_per_attempt = max(1, (deadline - 30 - time.time()) / 5)
        result = _try_rank(r, n, m, p, max_iter=200, seed=r, time_limit=time_per_attempt)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
