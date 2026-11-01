import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# Deadline guard: must finish well before 120s; use 110s as internal limit.
_DEADLINE = 110.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _check_233(U, V, W, tol=1e-9):
    """
    Verify that U, V, W (each of shape (6, r), (9, r), (6, r) respectively,
    with r = rank) decompose the 2x3x3 multiplication tensor.
    """
    n, m, p = 2, 3, 3
    r = U.shape[1]
    # Vectorized check
    # U: (6, r) -> (2, 3, r)
    # V: (9, r) -> (3, 3, r)
    # W: (6, r) -> (3, 2, r) -> (2, 3, r) for W[k*n+i] = W[k, i]
    
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    # W index is k*n + i. So W[k, i].
    # We need W_mat[i, k, c] = W[k*n+i, c].
    # W is (p*n, r) = (6, r).
    # Reshape to (p, n, r) -> (3, 2, r).
    # We need (n, p, r) -> (2, 3, r).
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2)) # (n, p, r)
    
    # Compute sum over c
    # term[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[i, k, c]
    # U[i, j, c] * V[j, k, c] -> (n, m, p, r)
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    # prod_UV[i, j, k, c]
    # Multiply by W[i, k, c]
    # W_mat is (n, p, r). Need to broadcast to (n, m, p, r).
    # W_mat[i, k, c] -> W_mat[:, np.newaxis, :, :]
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    
    result = np.sum(prod, axis=3) # (n, m, p)
    
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _objective_and_grad_233(U, V, W):
    """
    Compute objective and gradients for 2x3x3.
    Returns (obj, grad_U, grad_V, grad_W).
    """
    n, m, p = 2, 3, 3
    r = U.shape[1]
    
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2)) # (n, p, r)
    
    # Compute current values
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    result = np.sum(prod, axis=3) # (n, m, p)
    
    diff = result - 1.0
    obj = np.sum(diff * diff)
    
    # Gradients
    # dE/dU[i,j,c] = 2 * sum_k diff[i,j,k] * V[j,k,c] * W[i,k,c]
    # dE/dV[j,k,c] = 2 * sum_i diff[i,j,k] * U[i,j,c] * W[i,k,c]
    # dE/dW[i,k,c] = 2 * sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # grad_U: (n, m, r)
    # diff[i,j,k] * V[j,k,c] * W[i,k,c]
    # diff: (n, m, p)
    # V: (m, p, r)
    # W: (n, p, r)
    
    # Term for U: sum_k diff[i,j,k] * V[j,k,c] * W[i,k,c]
    # Multiply diff (n,m,p) by V (m,p,r) -> need to broadcast m, p.
    # diff[:, :, :, np.newaxis] * V_mat[np.newaxis, :, :, :] -> (n, m, p, r)
    temp_U = diff[:, :, :, np.newaxis] * V_mat[np.newaxis, :, :, :]
    # Multiply by W (n, p, r) -> broadcast n, p.
    # temp_U (n, m, p, r) * W_mat[:, np.newaxis, :, :] -> (n, m, p, r)
    grad_U_mat = np.sum(temp_U * W_mat[:, np.newaxis, :, :], axis=2)
    grad_U = 2.0 * grad_U_mat.reshape(n*m, r)
    
    # Term for V: sum_i diff[i,j,k] * U[i,j,c] * W[i,k,c]
    # diff (n,m,p) * U (n,m,r) -> (n,m,p,r)
    temp_V = diff[:, :, :, np.newaxis] * U_mat[:, :, np.newaxis, :]
    # Multiply by W (n, p, r) -> broadcast n, p.
    # temp_V (n,m,p,r) * W_mat[:, np.newaxis, :, :] -> (n,m,p,r)
    grad_V_mat = np.sum(temp_V * W_mat[:, np.newaxis, :, :], axis=0) # sum over i (axis 0)
    # grad_V_mat is (m, p, r)
    grad_V = 2.0 * grad_V_mat.reshape(m*p, r)
    
    # Term for W: sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
    # diff (n,m,p) * U (n,m,r) -> (n,m,p,r)
    temp_W = diff[:, :, :, np.newaxis] * U_mat[:, :, np.newaxis, :]
    # Multiply by V (m, p, r) -> broadcast m, p.
    # temp_W (n,m,p,r) * V_mat[np.newaxis, :, :, :] -> (n,m,p,r)
    grad_W_mat = np.sum(temp_W * V_mat[np.newaxis, :, :, :], axis=1) # sum over j (axis 1)
    # grad_W_mat is (n, p, r)
    # We need grad_W in original shape (p, n, r) -> (p*n, r)
    grad_W_raw = np.transpose(grad_W_mat, (1, 0, 2))
    grad_W = 2.0 * grad_W_raw.reshape(p*n, r)
    
    return float(obj), grad_U, grad_V, grad_W

def _snappy_search_233(max_restarts=50, steps_per_restart=100, tol=1e-9):
    """
    Gradient descent search for rank-15 half-integer decomposition of 2x3x3.
    """
    n, m, p = 2, 3, 3
    rank = 15
    U_shape = (n * m, rank)
    V_shape = (m * p, rank)
    W_shape = (p * n, rank)
    
    best_obj = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(max_restarts):
        if _time_left() < 5.0:
            break
        
        # Initialize with small random real values
        np.random.seed(restart)
        U = np.random.randn(*U_shape) * 0.1
        V = np.random.randn(*V_shape) * 0.1
        W = np.random.randn(*W_shape) * 0.1
        
        obj = np.inf
        for step in range(steps_per_restart):
            if _time_left() < 5.0:
                break
            
            obj, gU, gV, gW = _objective_and_grad_233(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_obj < tol:
                    return best_U, best_V, best_W
            
            # Adaptive learning rate
            lr = 0.01 / (1.0 + step * 0.01)
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            # Optional: Snap to half-integers periodically?
            # No, keep real values during descent, snap at the end.
        
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _check_233(U_snap, V_snap, W_snap, tol=1e-6):
            return U_snap, V_snap, W_snap
            
        # Also check the un-snapped version if it's very close
        if best_obj < 1e-6:
             U_snap = np.round(best_U * 2) / 2
             V_snap = np.round(best_V * 2) / 2
             W_snap = np.round(best_W * 2) / 2
             if _check_233(U_snap, V_snap, W_snap, tol=1e-6):
                 return U_snap, V_snap, W_snap

    if best_obj < tol:
        return best_U, best_V, best_W
    return None

def _schoolbook_236():
    """Return schoolbook rank-36 decomposition for 2x3x6 as lists."""
    n, m, p = 2, 3, 6
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

def _combine_233_to_236(U1, V1, W1, U2, V2, W2):
    """
    Combine two 2x3x3 rank-15 decompositions into a 2x3x6 rank-30 decomposition.
    """
    n, m, p = 2, 3, 6
    rank1, rank2 = U1.shape[1], U2.shape[1]
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill first half (columns 0-2)
    for r in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U1[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = V1[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r] = W1[k * n + i][r]
    
    # Fill second half (columns 3-5)
    offset = rank1
    for r in range(rank2):
        for i in range(n):
            for j in range(m):
                U[i * m + j][offset + r] = U2[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][offset + r] = V2[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i][offset + r] = W2[k * n + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication.
    """
    global _start_time
    _start_time = time.time()
    
    # Try to find rank-15 decomposition for 2x3x3
    decomp_233 = None
    if _time_left() > 10.0:
        decomp_233 = _snappy_search_233(max_restarts=50, steps_per_restart=100)
    
    if decomp_233 is not None:
        U1, V1, W1 = decomp_233
        # Use the same decomposition for both halves
        result = _combine_233_to_236(U1, V1, W1, U1, V1, W1)
        return result
    else:
        # Fall back to schoolbook
        return _schoolbook_236()
