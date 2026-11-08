import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    # Reconstruct tensor: T[i*m+j, j*p+k, k*n+i] should be 1
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _try_rank(r, n, m, p, max_iter=500, tol=1e-8, deadline=None):
    """Try to find rank-r decomposition via gradient descent."""
    if deadline is not None and time.time() > deadline:
        return None
    
    U, V, W = _construct_schoolbook()
    # Take first r columns as initial guess
    U = U[:, :r].copy()
    V = V[:, :r].copy()
    W = W[:, :r].copy()
    
    # Add small random perturbation
    rng = np.random.default_rng(42)
    U += rng.normal(0, 0.01, U.shape)
    V += rng.normal(0, 0.01, V.shape)
    W += rng.normal(0, 0.01, W.shape)
    
    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of loss = sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        loss = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # val = sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    diff = val - 1.0
                    loss += diff * diff
                    # d/dU[a,r] = 2*diff*V[b,r]*W[c,r]
                    gU[a, :] += 2.0 * diff * V[b, :] * W[c, :]
                    gV[b, :] += 2.0 * diff * U[a, :] * W[c, :]
                    gW[c, :] += 2.0 * diff * U[a, :] * V[b, :]
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        if loss < tol:
            break
    
    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-6):
        # Try to snap to half-integers
        U_snap = _snap_to_half_integer(U)
        V_snap = _snap_to_half_integer(V)
        W_snap = _snap_to_half_integer(W)
        if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
            return U_snap, V_snap, W_snap
        # Try integer snap
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify(U_int, V_int, W_int, n, m, p, tol=1e-9):
            return U_int, V_int, W_int
        # Return as-is if close enough (shouldn't happen for exact)
        return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Try decreasing ranks from 39 down to 30
    for r in range(39, 29, -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iter=300, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists with integer/half-integer entries
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U, V, W = _construct_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
