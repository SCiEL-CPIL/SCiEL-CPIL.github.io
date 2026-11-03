"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _snap_to_half_int(x, tol=1e-9):
    """Snap value to nearest half-integer if within tolerance."""
    snapped = np.round(x * 2) / 2
    if np.abs(x - snapped) < tol:
        return snapped
    return x

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_update_U(U, V, W, n, m, p, r):
    """Update U given V and W."""
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for r_idx in range(r):
                # U[u_idx, r_idx] = sum_k V[j*p+k, r_idx] * W[k*n+i, r_idx] / norm
                num = 0.0
                den = 0.0
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    vw = V[v_idx, r_idx] * W[w_idx, r_idx]
                    num += 1.0 * vw  # T[i,j,k] = 1
                    den += vw * vw
                if den > 1e-12:
                    U_new[u_idx, r_idx] = num / den
                else:
                    U_new[u_idx, r_idx] = 0.0
    return U_new

def _als_update_V(U, V, W, n, m, p, r):
    """Update V given U and W."""
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            for r_idx in range(r):
                num = 0.0
                den = 0.0
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    uw = U[u_idx, r_idx] * W[w_idx, r_idx]
                    num += 1.0 * uw  # T[i,j,k] = 1
                    den += uw * uw
                if den > 1e-12:
                    V_new[v_idx, r_idx] = num / den
                else:
                    V_new[v_idx, r_idx] = 0.0
    return V_new

def _als_update_W(U, V, W, n, m, p, r):
    """Update W given U and V."""
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            for r_idx in range(r):
                num = 0.0
                den = 0.0
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    uv = U[u_idx, r_idx] * V[v_idx, r_idx]
                    num += 1.0 * uv  # T[i,j,k] = 1
                    den += uv * uv
                if den > 1e-12:
                    W_new[w_idx, r_idx] = num / den
                else:
                    W_new[w_idx, r_idx] = 0.0
    return W_new

def _try_rank_als(r, n, m, p, max_iters=300, seed=42):
    """Try to find rank-r decomposition using ALS with half-integer snapping."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    best_err = float('inf')
    best_result = None
    
    for iteration in range(max_iters):
        U = _als_update_U(U, V, W, n, m, p, r)
        V = _als_update_V(U, V, W, n, m, p, r)
        W = _als_update_W(U, V, W, n, m, p, r)
        
        # Snap to half-integers
        U = np.vectorize(_snap_to_half_int)(U)
        V = np.vectorize(_snap_to_half_int)(V)
        W = np.vectorize(_snap_to_half_int)(W)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-12:
            U, V, W = best_result
            if _verify_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
            break
    
    if best_result is not None:
        U, V, W = best_result
        if _verify_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try ranks from low to high
    for r in range(8, 25):
        if time.time() > deadline:
            break
        for seed in range(15):
            if time.time() > deadline:
                break
            result = _try_rank_als(r, n, m, p, max_iters=150, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _try_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
